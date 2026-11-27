import io
import json
import subprocess
import sys

import pytest
from scripts.g5k import chain, clusters, run_log, status

A4500 = {"gpu_devices": {"0": {"model": "NVIDIA RTX A4500", "memory": 20480}}}


def write_metrics(path, *records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


# chain: the decision each training link takes before it starts


def test_the_first_link_starts_with_no_failures():
    assert chain.next_state({}, 0) == {"step": 0, "failures": 0}


def test_a_link_that_moved_on_resets_the_failures():
    state = {"step": 100, "failures": 2}

    assert chain.next_state(state, 150) == {"step": 150, "failures": 0}


def test_a_link_that_did_not_move_on_adds_a_failure():
    state = {"step": 100, "failures": 1}

    assert chain.next_state(state, 100) == {"step": 100, "failures": 2}


def test_begin_runs_until_the_chain_has_failed_three_times_in_a_row(tmp_path):
    run, state = tmp_path / "runs" / "shuffle", tmp_path / "state" / "shuffle.json"
    write_metrics(run / "metrics.jsonl", {"step": 0})

    verdicts = [chain.begin(run, state) for _ in range(4)]

    assert verdicts == ["run", "run", "run", "stop"]
    assert json.loads(state.read_text()) == {"step": 0, "failures": 3}


def test_begin_lets_the_chain_go_on_after_progress(tmp_path):
    run, state = tmp_path / "shuffle", tmp_path / "state.json"
    write_metrics(run / "metrics.jsonl", {"step": 0})
    chain.begin(run, state)
    chain.begin(run, state)
    write_metrics(run / "metrics.jsonl", {"step": 0}, {"step": 200})

    assert chain.begin(run, state) == "run"
    assert json.loads(state.read_text()) == {"step": 200, "failures": 0}


def test_begin_respects_the_failure_limit_it_is_given(tmp_path):
    run, state = tmp_path / "shuffle", tmp_path / "state.json"

    assert chain.begin(run, state, max_failures=1) == "run"
    assert chain.begin(run, state, max_failures=1) == "stop"


def test_chain_main_prints_the_verdict(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["chain.py", "begin", str(tmp_path / "run"), str(tmp_path / "state.json")]
    )

    chain.main()

    assert capsys.readouterr().out == "run\n"


# clusters: which clusters a one-GPU job can use


def test_vram_is_read_in_bytes_mib_or_gb():
    assert clusters.vram_gb(48_000_000_000) == 48
    assert clusters.vram_gb(20480) == 20
    assert clusters.vram_gb(24) == 24


def test_a_production_ampere_cluster_with_enough_memory_is_usable():
    found = clusters.usable({"uid": "gros", "queues": ["abaca"]}, [A4500], free_nodes=3)

    assert found == clusters.Cluster("gros", "NVIDIA RTX A4500", 20.0, 3)


@pytest.mark.parametrize(
    ("cluster", "nodes"),
    [
        ({"uid": "x", "queues": ["default"]}, [A4500]),
        ({"uid": "x", "queues": ["abaca"], "exotic": True}, [A4500]),
        ({"uid": "x", "queues": ["abaca"]}, [{"gpu_devices": {}}]),
        ({"uid": "x", "queues": ["abaca"]}, [{}]),
        ({"uid": "x", "queues": ["abaca"]}, [{"gpu_devices": {"0": {"model": "Tesla V100"}}}]),
        ({"uid": "x", "queues": ["abaca"]}, [{"gpu_devices": {"0": {"model": "GH200"}}}]),
        (
            {"uid": "x", "queues": ["abaca"]},
            [{"gpu_devices": {"0": {"model": "NVIDIA RTX 2080", "memory": 8192}}}],
        ),
        (
            {"uid": "x", "queues": ["abaca"]},
            [{"gpu_devices": {"0": {"model": "NVIDIA A100", "memory": 8192}}}],
        ),
    ],
)
def test_clusters_that_a_one_gpu_job_cannot_use_are_skipped(cluster, nodes):
    assert clusters.usable(cluster, nodes, free_nodes=1) is None


def test_free_nodes_are_counted_per_cluster():
    status_payload = {
        "nodes": {
            "gros-1": {"soft": "free"},
            "gros-2": {"soft": "busy"},
            "gros-3": {"soft": "free"},
            "ecotype-9": {"soft": "free"},
        }
    }

    assert clusters.free_nodes_by_cluster(status_payload) == {"gros": 2, "ecotype": 1}


def test_free_nodes_accept_a_status_without_the_nodes_key():
    assert clusters.free_nodes_by_cluster({"gros-1": {"soft": "free"}}) == {"gros": 1}


def test_usable_clusters_are_listed_most_free_first():
    listing = [
        {"uid": "a", "queues": ["abaca"]},
        {"uid": "b", "queues": ["abaca"]},
        {"uid": "c", "queues": ["default"]},
    ]
    nodes = {"a": [A4500], "b": [A4500], "c": [A4500]}

    found = clusters.usable_clusters(listing, nodes, {"a": 1, "b": 4})

    assert [cluster.uid for cluster in found] == ["b", "a"]


def test_fetch_combines_the_api_answers(monkeypatch):
    answers = {
        "clusters": [{"uid": "gros", "queues": ["abaca"]}],
        "gros/nodes": {"items": [A4500]},
        "status": {"nodes": {"gros-1": {"soft": "free"}}},
    }

    def fake_get(url):
        if url.endswith("/status?disks=no&job_details=no"):
            return answers["status"]
        if url.endswith("/nodes"):
            return answers["gros/nodes"]
        return answers["clusters"]

    monkeypatch.setattr(clusters, "_get", fake_get)

    assert [cluster.uid for cluster in clusters.fetch("nancy")] == ["gros"]


def test_get_decodes_the_json_answer(monkeypatch):
    monkeypatch.setattr(
        clusters.urllib.request, "urlopen", lambda _url, **_options: io.BytesIO(b'[{"uid": "a"}]')
    )

    assert clusters._get("https://example.invalid/clusters") == [{"uid": "a"}]


def test_items_accept_a_list_or_an_items_envelope():
    assert clusters._items([{"uid": "a"}]) == [{"uid": "a"}]
    assert clusters._items({"items": [{"uid": "a"}]}) == [{"uid": "a"}]


def test_clusters_main_lists_the_usable_ones(monkeypatch, capsys):
    found = [clusters.Cluster("gros", "NVIDIA RTX A4500", 20.0, 3)]
    monkeypatch.setattr(clusters, "fetch", lambda _site: found)
    monkeypatch.setattr("sys.argv", ["clusters.py", "--site", "nancy"])

    clusters.main()

    assert "gros" in capsys.readouterr().out


def test_clusters_main_says_when_none_is_usable(monkeypatch, capsys):
    monkeypatch.setattr(clusters, "fetch", lambda _site: [])
    monkeypatch.setattr("sys.argv", ["clusters.py", "--site", "lyon"])

    clusters.main()

    assert "no usable abaca cluster" in capsys.readouterr().out


# status: the chains of this site as OAR and the metrics see them


def test_live_jobs_skip_the_ended_ones():
    listing = {
        "1": {"Job_Id": 1, "name": "gpt2-shuffle-gros", "state": "Running"},
        "2": {"Job_Id": 2, "name": "gpt2-shuffle-gros", "state": "Terminated"},
        "3": {"Job_Id": 3, "name": "gpt2-prep-gros", "state": "Error"},
        "4": {"Job_Id": 4, "name": "gpt2-cooldown-gros", "state": "Waiting"},
    }

    jobs = status.live_jobs(listing)

    assert [job["Job_Id"] for job in jobs] == [1, 4]


def test_live_jobs_accept_a_list():
    assert status.live_jobs([{"name": "a", "state": "Waiting"}]) == [
        {"name": "a", "state": "Waiting"}
    ]


def test_status_reads_no_records_without_a_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step": 2')

    assert run_log.read_records(path) == [{"step": 1}]
    assert run_log.read_records(tmp_path / "missing.jsonl") == []


def test_last_validation_is_the_latest_loss_or_none():
    assert status.last_validation([{"step": 1}]) is None
    assert status.last_validation([{"val_loss": 4.1}, {"step": 2}, {"val_loss": 4.0}]) == 4.0


def test_busy_matches_the_name_prefixes():
    jobs = [{"name": "gpt2-shuffle-gros"}, {"name": "other"}, {"name": None}]

    assert status.busy(jobs, ["gpt2-shuffle-"]) == [{"name": "gpt2-shuffle-gros"}]


def test_job_ids_fall_back_to_the_id_key_then_a_placeholder():
    assert status.job_id({"Job_Id": 7}) == "7"
    assert status.job_id({"id": 8}) == "8"
    assert status.job_id({}) == "?"


def test_report_shows_each_arm_and_the_live_jobs(tmp_path):
    write_metrics(
        tmp_path / "runs" / "init42-data42" / "shuffle" / "metrics.jsonl",
        {"step": 0},
        {"step": 1716, "val_loss": 4.001},
    )
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "shuffle-init42-data42.done").write_text("")
    write_metrics(tmp_path / "runs" / "init43-data42" / "cooldown" / "metrics.jsonl", {"step": 300})
    jobs = [{"Job_Id": 9, "name": "gpt2-cooldown-gros", "state": "Waiting"}]

    text = status.report(tmp_path / "runs", jobs, tmp_path / "state")

    assert "shuffle-init42-data42" in text
    assert "step  1716  val_loss  4.001  published" in text
    assert "cooldown-init43-data42" in text
    assert "step   300" in text
    assert "job 9  gpt2-cooldown-gros  Waiting" in text


def test_report_says_when_there_are_no_runs(tmp_path):
    assert status.report(tmp_path / "missing", [], tmp_path) == "no runs yet"


def test_oarstat_jobs_reads_the_user_listing(monkeypatch):
    listing = {"1": {"name": "gpt2-shuffle-gros", "state": "Running"}}
    seen = {}

    def fake_run(command, **_options):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(listing))

    monkeypatch.setattr(status.subprocess, "run", fake_run)

    assert status.oarstat_jobs() == [{"name": "gpt2-shuffle-gros", "state": "Running"}]
    assert seen["command"] == ["oarstat", "-u", "-J"]


def test_oarstat_jobs_is_empty_without_output(monkeypatch):
    monkeypatch.setattr(
        status.subprocess,
        "run",
        lambda command, **_options: subprocess.CompletedProcess(command, 0, stdout=""),
    )

    assert status.oarstat_jobs() == []


def test_status_main_exits_1_when_a_prefix_is_busy(monkeypatch, capsys):
    monkeypatch.setattr(status, "oarstat_jobs", lambda: [{"Job_Id": 5, "name": "gpt2-prep-x"}])
    monkeypatch.setattr("sys.argv", ["status.py", "--busy", "gpt2-prep-"])

    with pytest.raises(SystemExit) as exit_info:
        status.main()

    assert exit_info.value.code == 1
    assert "live job 5: gpt2-prep-x" in capsys.readouterr().out


def test_status_main_exits_0_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(status, "oarstat_jobs", list)
    monkeypatch.setattr("sys.argv", ["status.py", "--busy", "gpt2-prep-"])

    with pytest.raises(SystemExit) as exit_info:
        status.main()

    assert exit_info.value.code == 0


def test_status_main_prints_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(status, "oarstat_jobs", list)
    monkeypatch.setattr("sys.argv", ["status.py", "--runs", str(tmp_path / "none")])

    status.main()

    assert capsys.readouterr().out == "no runs yet\n"


# publish: send a finished arm to the model repository, then delete its weights from the site


def test_publish_uploads_the_finished_arm_and_removes_its_weights(tmp_path, monkeypatch, capsys):
    from scripts.g5k import publish

    out = tmp_path / "runs" / "init42-data42" / "shuffle"
    out.mkdir(parents=True)
    (out / "checkpoint.pt").write_bytes(b"state")
    (out / "model.pt").write_bytes(b"weights")
    (out / "config.json").write_text("{}")
    sent = {}
    monkeypatch.setattr(publish, "HfApi", lambda: "api")
    monkeypatch.setattr(
        publish,
        "publish_final",
        lambda api, repo, run, folder: sent.update(api=api, repo=repo, run=run, folder=folder),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish.py",
            "--arm",
            "shuffle",
            "--seed",
            "42",
            "--out",
            str(out),
            "--repo",
            "user/repo",
        ],
    )

    publish.main()

    assert sent == {
        "api": "api",
        "repo": "user/repo",
        "run": "init42-data42/shuffle",
        "folder": out,
    }
    assert not (out / "checkpoint.pt").exists()
    assert not (out / "model.pt").exists()
    assert not (out / "config.json").exists()
    assert "published init42-data42/shuffle to user/repo" in capsys.readouterr().out


# dashboard: rebuild the Trackio dashboard from every run, without overwriting anything silently


def test_fetch_reads_a_bucket_file_with_the_token(monkeypatch):
    from scripts.g5k import dashboard

    seen = {}

    def fake_urlopen(request, timeout):
        assert timeout == 120
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        return io.BytesIO(b"content")

    monkeypatch.setattr(dashboard.urllib.request, "urlopen", fake_urlopen)

    assert dashboard.fetch("user/bucket", "runs.json", "tok") == b"content"
    assert seen == {
        "url": "https://huggingface.co/buckets/user/bucket/resolve/runs.json",
        "auth": "Bearer tok",
    }


def test_backup_copies_each_dashboard_file_under_a_timestamped_folder(monkeypatch):
    from scripts.g5k import dashboard

    sent = {}

    class FakeApi:
        def batch_bucket_files(self, bucket, add, token):
            sent.update(bucket=bucket, add=add, token=token)

    monkeypatch.setattr(dashboard, "fetch", lambda _bucket, name, _token: f"{name}!".encode())
    monkeypatch.setattr(dashboard, "HfApi", FakeApi)

    dashboard.backup_bucket("user/bucket", "tok", "20260101T000000Z")

    assert sent["bucket"] == "user/bucket"
    assert sent["token"] == "tok"
    assert sent["add"] == [
        (f"{name}!".encode(), f"backups/20260101T000000Z/{name}")
        for name in dashboard.DASHBOARD_FILES
    ]


def test_metric_values_skip_identifiers_and_missing_values():
    from scripts.g5k import dashboard

    row = {
        "id": 1,
        "run_id": "r",
        "run_name": "shuffle-init42-data42",
        "step": 50,
        "val_loss": 4.0,
        "train_loss": float("nan"),
        "lr": None,
        "tokens_per_s": 9.0,
    }

    assert dashboard._metrics_of(row) == {"val_loss": 4.0, "tokens_per_s": 9.0}


def test_bucket_runs_rebuild_each_run_from_the_bucket_files():
    from scripts.g5k import dashboard

    runs_json = [{"name": "shuffle-init42-data42"}, {"name": "cooldown-init42-data42"}]
    configs = [
        {
            "run_name": "shuffle-init42-data42",
            "arm": "shuffle",
            "train": {"seed": 42},
            "model": {"n": 1},
        },
        {
            "run_name": "cooldown-init42-data42",
            "arm": "cooldown",
            "train": {"seed": 42},
            "model": {"n": 1},
        },
    ]
    metrics = [
        {
            "run_name": "shuffle-init42-data42",
            "step": 100,
            "train_loss": 3.0,
            "run_id": "a",
            "id": 2,
        },
        {"run_name": "shuffle-init42-data42", "step": 50, "val_loss": 4.0, "run_id": "a", "id": 1},
        {"run_name": "shuffle-init42-data42", "step": 60, "train_loss": None},
    ]

    runs = dashboard.bucket_runs(runs_json, configs, metrics)

    assert runs[0] == dashboard.Run(
        "shuffle-init42-data42",
        {"arm": "shuffle", "train": {"seed": 42}, "model": {"n": 1}},
        [(50, {"val_loss": 4.0}), (100, {"train_loss": 3.0})],
    )
    assert runs[1].records == []


def test_site_runs_read_each_arm_of_a_seed_and_skip_a_cut_line(tmp_path):
    from scripts.g5k import dashboard

    write_metrics(
        tmp_path / "init43-data42" / "shuffle" / "metrics.jsonl", {"step": 50, "val_loss": 4.0}
    )
    with (tmp_path / "init43-data42" / "shuffle" / "metrics.jsonl").open("a") as file:
        file.write('{"step": 100, "tra')
    write_metrics(tmp_path / "init43-data42" / "ascending" / "metrics.jsonl", {"step": 0})
    (tmp_path / "init43-data42" / "empty").mkdir()

    runs = dashboard.site_runs(tmp_path, 43)

    assert [run.name for run in runs] == ["ascending-init43-data42", "shuffle-init43-data42"]
    shuffle = runs[1]
    assert shuffle.config["arm"] == "shuffle"
    assert shuffle.config["train"]["seed"] == 43
    assert shuffle.records == [(50, {"val_loss": 4.0})]


def test_merged_replaces_a_run_the_site_also_has_and_keeps_the_rest():
    from scripts.g5k import dashboard

    bucket = [
        dashboard.Run("shuffle-init42-data42", {}, []),
        dashboard.Run("shuffle-init43-data42", {"v": 1}, []),
    ]
    sites = [
        dashboard.Run("shuffle-init43-data42", {"v": 2}, []),
        dashboard.Run("ascending-init43-data42", {}, []),
    ]

    names = [(run.name, run.config) for run in dashboard.merged(bucket, sites)]

    assert names == [
        ("shuffle-init42-data42", {}),
        ("shuffle-init43-data42", {"v": 2}),
        ("ascending-init43-data42", {}),
    ]


def test_write_dashboard_logs_every_run_then_syncs_once(monkeypatch):
    from scripts.g5k import dashboard

    calls = []

    class FakeTrackio:
        def init(self, project, name, group, config):
            calls.append(("init", project, name, group, config))

        def log(self, values, step):
            calls.append(("log", values, step))

        def finish(self):
            calls.append(("finish",))

        def sync(self, project, space_id, bucket_id, sdk, private):
            calls.append(("sync", project, space_id, bucket_id, sdk, private))

    monkeypatch.setitem(sys.modules, "trackio", FakeTrackio())
    run = dashboard.Run("shuffle-init43-data42", {"arm": "shuffle"}, [(50, {"val_loss": 4.0})])

    dashboard.write_dashboard([run], "proj", "space", "bucket")

    assert calls == [
        ("init", "proj", "shuffle-init43-data42", "shuffle", {"arm": "shuffle"}),
        ("log", {"val_loss": 4.0}, 50),
        ("finish",),
        ("sync", "proj", "space", "bucket", "static", False),
    ]


def stub_rebuild(monkeypatch, events):
    """Stub the bucket and the writing of a rebuild, recording their calls in events."""
    from scripts.g5k import dashboard

    monkeypatch.setattr(dashboard, "get_token", lambda: "tok")
    monkeypatch.setattr(
        dashboard, "backup_bucket", lambda _bucket, _token, stamp: events.append(("backup", stamp))
    )
    monkeypatch.setattr(
        dashboard,
        "fetch",
        lambda _bucket, _name, _token: json.dumps([{"name": "cooldown-init42-data42"}]).encode(),
    )
    tables = {
        "aux/configs.parquet": [
            {"run_name": "cooldown-init42-data42", "arm": "cooldown", "train": {}, "model": {}}
        ],
        "metrics.parquet": [{"run_name": "cooldown-init42-data42", "step": 0, "val_loss": 5.0}],
    }
    monkeypatch.setattr(dashboard, "_read_parquet", lambda _bucket, name, _token: tables[name])
    monkeypatch.setattr(
        dashboard,
        "write_dashboard",
        lambda runs, _project, _space, _bucket: events.append(
            ("write", [run.name for run in runs])
        ),
    )


def test_dashboard_main_backs_up_then_rebuilds_with_the_site_runs(monkeypatch, tmp_path, capsys):
    from scripts.g5k import dashboard

    write_metrics(
        tmp_path / "init43-data42" / "shuffle" / "metrics.jsonl", {"step": 50, "val_loss": 4.0}
    )
    events = []
    stub_rebuild(monkeypatch, events)
    monkeypatch.setattr("sys.argv", ["dashboard.py", "--runs", str(tmp_path), "--seeds", "43"])

    dashboard.main()

    assert events[0][0] == "backup"
    assert events[1] == ("write", ["cooldown-init42-data42", "shuffle-init43-data42"])
    assert "dashboard rebuilt with 2 runs" in capsys.readouterr().out


def test_dashboard_main_without_backup_rebuilds_under_the_site_lock(monkeypatch, tmp_path, capsys):
    from scripts.g5k import dashboard

    write_metrics(
        tmp_path / "init43-data42" / "shuffle" / "metrics.jsonl", {"step": 50, "val_loss": 4.0}
    )
    events = []
    stub_rebuild(monkeypatch, events)
    monkeypatch.setattr(
        "sys.argv", ["dashboard.py", "--runs", str(tmp_path), "--seeds", "43", "--no-backup"]
    )

    dashboard.main()

    assert events == [("write", ["cooldown-init42-data42", "shuffle-init43-data42"])]
    assert (tmp_path / ".dashboard.lock").exists()
    assert "dashboard rebuilt with 2 runs\n" in capsys.readouterr().out


class FakeChild:
    """A rebuild process that the test finishes by hand: its returncode is None while it runs."""

    def __init__(self, command, env):
        self.command = command
        self.env = env
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


def fake_rebuilds(monkeypatch, tmp_path):
    """Replace the rebuild processes with FakeChild objects, and return the list they join."""
    from scripts.g5k import dashboard

    children = []

    def popen(command, env, **_options):
        child = FakeChild(command, env)
        children.append(child)
        return child

    monkeypatch.setattr(dashboard.subprocess, "Popen", popen)
    monkeypatch.setattr(dashboard.tempfile, "mkdtemp", lambda prefix: str(tmp_path / prefix))
    return children


def test_site_dashboard_rebuilds_the_seed_at_every_fifty_training_steps(monkeypatch, tmp_path):
    from scripts.g5k import dashboard

    children = fake_rebuilds(monkeypatch, tmp_path)
    tracker = dashboard.SiteDashboard(43, runs=tmp_path)

    tracker.log({"train_loss": 5.0}, step=49)
    tracker.log({"val_loss": 4.0}, step=50)
    tracker.log({"train_loss": 4.5}, step=50)

    assert len(children) == 1
    assert children[0].command == [
        sys.executable,
        "-m",
        "scripts.g5k.dashboard",
        "--runs",
        str(tmp_path),
        "--seeds",
        "43",
        "--data-seed",
        "42",
        "--no-backup",
    ]
    assert children[0].env["TRACKIO_DIR"] == str(tmp_path / "trackio-")


def test_site_dashboard_skips_a_rebuild_still_running_and_reports_a_failed_one(
    monkeypatch, tmp_path, capsys
):
    from scripts.g5k import dashboard

    children = fake_rebuilds(monkeypatch, tmp_path)
    tracker = dashboard.SiteDashboard(43, runs=tmp_path)

    tracker.log({"train_loss": 4.5}, step=50)
    tracker.log({"train_loss": 4.4}, step=100)  # the rebuild of step 50 is still running
    assert len(children) == 1

    children[0].returncode = 1  # it failed: reported, then the next rebuild starts
    tracker.log({"train_loss": 4.3}, step=150)
    tracker.finish()  # waits for the last rebuild, then publishes the final state

    assert len(children) == 3
    assert "dashboard publish deferred (exit 1) at step 50" in capsys.readouterr().out


def test_site_dashboard_publishes_the_final_state_when_the_run_ends(monkeypatch, tmp_path):
    from scripts.g5k import dashboard

    children = fake_rebuilds(monkeypatch, tmp_path)

    dashboard.SiteDashboard(43, runs=tmp_path).finish()

    assert [child.command[-1] for child in children] == ["--no-backup"]


def test_read_parquet_turns_a_bucket_table_into_rows(monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.g5k import dashboard

    buffer = io.BytesIO()
    pq.write_table(pa.table({"run_name": ["shuffle-init42-data42"], "step": [50]}), buffer)
    monkeypatch.setattr(dashboard, "fetch", lambda _bucket, _name, _token: buffer.getvalue())

    assert dashboard._read_parquet("user/bucket", "metrics.parquet", None) == [
        {"run_name": "shuffle-init42-data42", "step": 50}
    ]


# run_log: one reader for a run's metrics file, shared by the chain, the status and the dashboard


def test_run_log_reads_complete_records_and_skips_a_cut_line(tmp_path):
    from scripts.g5k import run_log

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0}\n{"step": 50, "val_loss": 4.0}\n{"step": 100, "tra')

    assert run_log.read_records(path) == [{"step": 0}, {"step": 50, "val_loss": 4.0}]


def test_run_log_reads_nothing_without_a_file(tmp_path):
    from scripts.g5k import run_log

    assert run_log.read_records(tmp_path / "missing.jsonl") == []


def test_run_log_last_step_is_the_last_record_that_has_one():
    from scripts.g5k import run_log

    assert run_log.last_step([{"step": 0}, {"step": 50}, {"val_loss": 4.0}]) == 50
    assert run_log.last_step([{"val_loss": 4.0}]) == 0
    assert run_log.last_step([]) == 0


def test_package_modules_are_run_as_modules_so_their_imports_resolve():
    from pathlib import Path

    scripts = Path("scripts/g5k")
    for name in ("chain", "status"):
        for shell in scripts.glob("*.sh"):
            assert f"scripts/g5k/{name}.py" not in shell.read_text(), (shell.name, name)


def test_chains_on_counts_each_live_chain_once_per_cluster():
    jobs = [
        {"name": "gpt2-shuffle-init44-data42-kinovis", "state": "Running"},
        {"name": "gpt2-shuffle-init44-data42-kinovis", "state": "Waiting"},
        {"name": "gpt2-ascending-init44-data42-kinovis", "state": "Waiting"},
        {"name": "gpt2-cooldown-init43-data42-vercors9", "state": "Running"},
        {"name": "gpt2-prep-kinovis", "state": "Error"},
    ]

    assert status.chains_on(jobs, "kinovis") == 2
    assert status.chains_on(jobs, "vercors9") == 1


def test_status_prints_the_chain_count_for_a_cluster(monkeypatch, capsys):
    monkeypatch.setattr(
        status,
        "oarstat_jobs",
        lambda: [{"name": "gpt2-shuffle-init44-data42-kinovis", "state": "Running"}],
    )
    monkeypatch.setattr("sys.argv", ["status.py", "--chains", "kinovis"])

    status.main()

    assert capsys.readouterr().out == "1\n"

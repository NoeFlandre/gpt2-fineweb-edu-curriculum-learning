import io
import json
import subprocess

import pytest
from scripts.g5k import chain, clusters, status

A4500 = {"gpu_devices": {"0": {"model": "NVIDIA RTX A4500", "memory": 20480}}}


def write_metrics(path, *records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


# chain: the decision each training link takes before it starts


def test_last_step_is_zero_without_metrics(tmp_path):
    assert chain.last_step(tmp_path / "metrics.jsonl") == 0


def test_last_step_reads_the_last_record(tmp_path):
    write_metrics(tmp_path / "metrics.jsonl", {"step": 0}, {"step": 50, "val_loss": 4.2})

    assert chain.last_step(tmp_path / "metrics.jsonl") == 50


def test_last_step_skips_a_line_cut_short_and_a_record_without_a_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 100}\n{"train_loss": 3.0}\n{"step": 15')

    assert chain.last_step(path) == 100


def test_last_step_is_zero_when_no_record_has_a_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"train_loss": 3.0}\n')

    assert chain.last_step(path) == 0


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


def test_vram_is_read_in_mib_or_gb():
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


def test_records_skip_a_line_cut_short(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step": 2')

    assert status.records(path) == [{"step": 1}]
    assert status.records(tmp_path / "missing.jsonl") == []


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
        tmp_path / "runs" / "shuffle" / "metrics.jsonl",
        {"step": 0},
        {"step": 1716, "val_loss": 4.001},
    )
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "shuffle.done").write_text("")
    write_metrics(tmp_path / "runs" / "cooldown" / "metrics.jsonl", {"step": 300})
    (tmp_path / "runs" / "empty").mkdir()
    jobs = [{"Job_Id": 9, "name": "gpt2-cooldown-gros", "state": "Waiting"}]

    text = status.report(tmp_path / "runs", jobs, tmp_path / "state")

    assert "shuffle" in text
    assert "step  1716  val_loss  4.001  published" in text
    assert "cooldown" in text
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

    out = tmp_path / "runs" / "shuffle"
    out.mkdir(parents=True)
    (out / "checkpoint.pt").write_bytes(b"state")
    (out / "model.pt").write_bytes(b"weights")
    (out / "config.json").write_text("{}")
    sent = {}
    monkeypatch.setattr(publish, "HfApi", lambda: "api")
    monkeypatch.setattr(
        publish,
        "publish_final",
        lambda api, repo, arm, folder: sent.update(api=api, repo=repo, arm=arm, folder=folder),
    )
    monkeypatch.setattr(
        "sys.argv", ["publish.py", "--arm", "shuffle", "--out", str(out), "--repo", "user/repo"]
    )

    publish.main()

    assert sent == {"api": "api", "repo": "user/repo", "arm": "shuffle", "folder": out}
    assert not (out / "checkpoint.pt").exists()
    assert not (out / "model.pt").exists()
    assert not (out / "config.json").exists()
    assert "published shuffle to user/repo" in capsys.readouterr().out

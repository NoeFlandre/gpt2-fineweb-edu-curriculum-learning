"""List the Grid'5000 clusters a one-GPU training job can use, with their free nodes.

A cluster is usable when it is in the production queue (abaca), is not exotic, has an Ampere or
newer NVIDIA GPU (bf16 autocast needs compute capability 8.0+), and has at least MIN_VRAM_GB of
memory per GPU. The Reference API is readable from a frontend only.

Usage (on a frontend): python3 scripts/g5k/clusters.py --site nancy
"""

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

API = "https://api.grid5000.fr/stable/sites/{site}"
QUEUE = "abaca"
MIN_VRAM_GB = 16
AMPERE_OR_NEWER = re.compile(
    r"\b(A100|A40|A30|A16|A10|A2|A4500|A5000|A6000|L4|L40S?|H100|H200|GH100)\b"
    r"|RTX A\d{3,4}|RTX (3|4|5)\d{3}|RTX [0-9]000 Ada|RTX PRO"
)


@dataclass(frozen=True)
class Cluster:
    """A cluster that a training job can be placed on."""

    uid: str
    gpu_model: str
    vram_gb: float
    free_nodes: int


def vram_gb(memory: float) -> float:
    """The GPU memory in GB. The API reports bytes; MiB and GB values are also accepted."""
    if memory > 1e9:
        return memory / 1e9
    if memory > 1024:
        return memory / 1024
    return float(memory)


def usable(cluster: dict, nodes: list[dict], free_nodes: int) -> Cluster | None:
    """The cluster as a Cluster if a one-GPU job can use it, otherwise None."""
    queues = cluster.get("queues") or []
    if QUEUE not in queues or cluster.get("exotic"):
        return None
    devices = [device for node in nodes[:1] for device in (node.get("gpu_devices") or {}).values()]
    if not devices:
        return None
    model = str(devices[0].get("model", ""))
    memory = vram_gb(float(devices[0].get("memory", 0)))
    if not AMPERE_OR_NEWER.search(model) or memory < MIN_VRAM_GB:
        return None
    return Cluster(cluster["uid"], model, memory, free_nodes)


def free_nodes_by_cluster(status: dict) -> dict[str, int]:
    """Number of free nodes per cluster, from the site status (a node is free when soft is free)."""
    counts: dict[str, int] = {}
    for name, node in status.get("nodes", status).items():
        if node.get("soft") == "free":
            cluster = name.rsplit("-", 1)[0]
            counts[cluster] = counts.get(cluster, 0) + 1
    return counts


def usable_clusters(
    clusters: list[dict], nodes_by_cluster: dict[str, list[dict]], free: dict[str, int]
) -> list[Cluster]:
    """Every usable cluster of the site, most free nodes first."""
    found = [
        usable(cluster, nodes_by_cluster.get(cluster["uid"], []), free.get(cluster["uid"], 0))
        for cluster in clusters
    ]
    return sorted((c for c in found if c), key=lambda c: -c.free_nodes)


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _items(payload: dict | list) -> list[dict]:
    return payload if isinstance(payload, list) else payload.get("items", [])


def fetch(site: str) -> list[Cluster]:
    """Query the Reference API and the site status for the usable clusters of a site."""
    base = API.format(site=site)
    clusters = _items(_get(f"{base}/clusters"))
    nodes = {c["uid"]: _items(_get(f"{base}/clusters/{c['uid']}/nodes")) for c in clusters}
    status = _get(f"{base}/status?disks=no&job_details=no")
    return usable_clusters(clusters, nodes, free_nodes_by_cluster(status))


def main() -> None:
    """Print the usable clusters of a site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, help="site name, such as nancy")
    args = parser.parse_args()
    found = fetch(args.site)
    if not found:
        print(f"no usable {QUEUE} cluster with an Ampere-or-newer GPU at {args.site}")
    for cluster in found:
        print(
            f"{cluster.uid:<16} {cluster.gpu_model:<32} {cluster.vram_gb:>5.0f} GB"
            f"  free nodes: {cluster.free_nodes}"
        )


if __name__ == "__main__":
    main()

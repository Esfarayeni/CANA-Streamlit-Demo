from pathlib import Path

from metric_store import load_metric, store_metric


def test_metric_store_round_trips_complete_edge_sets(tmp_path: Path):
    database = tmp_path / "metrics.sqlite3"
    values = {(1, 2): 0.25, (2, 3): -0.5}

    store_metric("model-1", "correlation", values, (0.25, 0.5), path=database)

    assert load_metric("model-1", "correlation", path=database) == (values, (0.25, 0.5))


def test_metric_store_distinguishes_cached_empty_metric_from_cache_miss(tmp_path: Path):
    database = tmp_path / "metrics.sqlite3"
    store_metric("empty", "activity", {}, (0.0, 1.0), path=database)

    assert load_metric("empty", "activity", path=database) == ({}, (0.0, 1.0))
    assert load_metric("missing", "activity", path=database) is None

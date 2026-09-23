"""Persistent, versioned storage for precomputed network edge metrics."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from metrics import compute_correlation_metrics, compute_structural_metrics


# Bump this whenever the definition or serialization of a stored metric changes.
METRIC_CACHE_VERSION = 1
EFFECTIVENESS = "effectiveness"
ACTIVITY = "activity"
EXCESS = "excess"
CORRELATION = "correlation"
METRIC_CACHE_PATH = Path(__file__).resolve().parent / "metric_cache.sqlite3"


def _connect(path: Path = METRIC_CACHE_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metric_sets (
            model_key TEXT NOT NULL,
            cache_version INTEGER NOT NULL,
            metric TEXT NOT NULL,
            minimum REAL NOT NULL,
            maximum REAL NOT NULL,
            PRIMARY KEY (model_key, cache_version, metric)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metric_edges (
            model_key TEXT NOT NULL,
            cache_version INTEGER NOT NULL,
            metric TEXT NOT NULL,
            source_id,
            target_id,
            value REAL NOT NULL,
            PRIMARY KEY (model_key, cache_version, metric, source_id, target_id)
        )
        """
    )
    return connection


def load_metric(model_key: str, metric: str, path: Path = METRIC_CACHE_PATH) -> tuple[dict[tuple[Any, Any], float], tuple[float, float]] | None:
    """Load a complete metric set, returning ``None`` only for a cache miss."""
    try:
        with _connect(path) as connection:
            summary = connection.execute(
                "SELECT minimum, maximum FROM metric_sets WHERE model_key = ? AND cache_version = ? AND metric = ?",
                (model_key, METRIC_CACHE_VERSION, metric),
            ).fetchone()
            if summary is None:
                return None
            rows = connection.execute(
                "SELECT source_id, target_id, value FROM metric_edges WHERE model_key = ? AND cache_version = ? AND metric = ?",
                (model_key, METRIC_CACHE_VERSION, metric),
            ).fetchall()
    except sqlite3.Error:
        return None
    return {(source, target): float(value) for source, target, value in rows}, (float(summary[0]), float(summary[1]))


def store_metric(model_key: str, metric: str, edge_values: dict[tuple[Any, Any], float], bounds: tuple[float, float], path: Path = METRIC_CACHE_PATH) -> None:
    """Atomically replace one metric set. Cache failures never block the UI."""
    try:
        with _connect(path) as connection:
            key = (model_key, METRIC_CACHE_VERSION, metric)
            connection.execute(
                "DELETE FROM metric_edges WHERE model_key = ? AND cache_version = ? AND metric = ?", key
            )
            connection.execute(
                "INSERT OR REPLACE INTO metric_sets (model_key, cache_version, metric, minimum, maximum) VALUES (?, ?, ?, ?, ?)",
                (*key, float(bounds[0]), float(bounds[1])),
            )
            connection.executemany(
                "INSERT INTO metric_edges (model_key, cache_version, metric, source_id, target_id, value) VALUES (?, ?, ?, ?, ?, ?)",
                [(*key, source, target, float(value)) for (source, target), value in edge_values.items()],
            )
    except sqlite3.Error:
        return


def cached_metric(model_key: str, metric: str, factory: Callable[[], tuple[dict[tuple[Any, Any], float], tuple[float, float]]]) -> tuple[dict[tuple[Any, Any], float], tuple[float, float]]:
    """Fetch a metric set, calculating and persisting it only on a cache miss."""
    cached = load_metric(model_key, metric)
    if cached is not None:
        return cached
    values, bounds = factory()
    store_metric(model_key, metric, values, bounds)
    return values, bounds


def effectiveness_metric(bn: Any) -> tuple[dict[tuple[Any, Any], float], tuple[float, float]]:
    graph = bn.effective_graph()
    values = {(source, target): float(data.get("weight", 0.0)) for source, target, data in graph.edges(data=True)}
    return values, (0.0, 1.0)


def cached_effectiveness(model_key: str, bn: Any) -> tuple[dict[tuple[Any, Any], float], tuple[float, float]]:
    return cached_metric(model_key, EFFECTIVENESS, lambda: effectiveness_metric(bn))


def cached_structural_metrics(model_key: str, bn: Any) -> tuple[dict[tuple[Any, Any], float], tuple[float, float], dict[tuple[Any, Any], float], tuple[float, float]]:
    """Fetch Activity and Excess together because CANA calculates them together."""
    activity = load_metric(model_key, ACTIVITY)
    excess = load_metric(model_key, EXCESS)
    if activity is not None and excess is not None:
        return activity[0], activity[1], excess[0], excess[1]

    _, activity_values, activity_bounds, excess_values, excess_bounds = compute_structural_metrics(bn, include_excess=True)
    store_metric(model_key, ACTIVITY, activity_values, activity_bounds)
    store_metric(model_key, EXCESS, excess_values, excess_bounds)
    return activity_values, activity_bounds, excess_values, excess_bounds


def cached_correlation(model_key: str, bn: Any) -> tuple[dict[tuple[Any, Any], float], tuple[float, float]]:
    def calculate() -> tuple[dict[tuple[Any, Any], float], tuple[float, float]]:
        _, values, bounds = compute_correlation_metrics(bn)
        return values, bounds

    return cached_metric(model_key, CORRELATION, calculate)


def effective_graph_from_values(structural_graph: Any, edge_values: dict[tuple[Any, Any], float]) -> Any:
    """Rebuild CANA's effective graph from stored weights and cheap topology."""
    graph = structural_graph.copy()
    graph.remove_edges_from(list(graph.edges()))
    for (source, target), value in edge_values.items():
        graph.add_edge(source, target, weight=float(value))
    return graph

"""Populate the persistent metric database for every bundled model.

Run from the project root:
    python scripts/precompute_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metric_store import cached_correlation, cached_effectiveness, cached_structural_metrics  # noqa: E402
from model_data import build_model_registry, get_bn_display_name  # noqa: E402


def model_key(source_label: str, model_name: str) -> str:
    return f"builtin:{source_label}:{model_name}"


def main() -> None:
    registry, failed = build_model_registry()
    if failed:
        print(f"Skipped unavailable models: {', '.join(sorted(failed))}")

    for display_name, entry in sorted(registry.items(), key=lambda item: item[0].lower()):
        bn, source = entry["bn"], entry["source"]
        key = model_key(source, get_bn_display_name(bn, fallback=display_name))
        try:
            cached_effectiveness(key, bn)
            cached_structural_metrics(key, bn)
            cached_correlation(key, bn)
            print(f"Cached: {display_name}")
        except Exception as error:
            print(f"Failed: {display_name}: {error}")


if __name__ == "__main__":
    main()

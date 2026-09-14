#!/usr/bin/env python3
"""Refresh the checked-in public metadata for CANA's Cell Collective models.

This deliberately runs outside the Streamlit app.  It keeps the deployed app
fast and reproducible while preserving a live Cell Collective fallback for a
model that has not yet been catalogued.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

import requests
from cana.datasets.bio import load_all_cell_collective_models


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "model_metadata.json"
DASHBOARD_URL = "https://research.cellcollective.org/research/dashboard/"
MODEL_URL = "https://research.cellcollective.org/web/api/model/{}"


def normalize(value: object) -> str:
    # A few public catalogue titles carry a terminal period while the CANA
    # dataset name does not.
    return str(value).strip().lower().rstrip(".")


def model_name(model: object) -> str:
    return str(
        getattr(model, "name", None)
        or getattr(model, "_name", None)
        or "Unnamed Cell Collective Model"
    )


def dashboard_payload(page_text: str) -> dict:
    marker = 'const data = {"published":'
    start = page_text.find(marker)
    if start < 0:
        raise ValueError("Cell Collective catalogue payload was not found.")
    return json.JSONDecoder().raw_decode(page_text[start + len("const data = "):])[0]


def public_references(version: dict) -> list[dict]:
    references = version.get("referenceMap") or {}
    model_references = version.get("modelReferenceMap") or {}
    ordered = sorted(
        model_references.values(),
        key=lambda item: (item.get("position", float("inf")), item.get("referenceId", float("inf"))),
    )
    result = []
    for item in ordered:
        reference = references.get(str(item.get("referenceId")))
        if not reference:
            continue
        citation = unescape(str(reference.get("text") or reference.get("shortCitation") or ""))
        citation = re.sub(r"<[^>]+>", " ", citation)
        citation = re.sub(r"\s+", " ", citation).strip()
        title = unescape(str(
            reference.get("title") or reference.get("articleTitle") or reference.get("publicationTitle") or ""
        ))
        title = re.sub(r"<[^>]+>", " ", title)
        title = re.sub(r"\s+", " ", title).strip()
        pmid = str(reference.get("pmid") or "").strip()
        doi = str(reference.get("doi") or "").strip()
        result.append({
            "citation": citation or "Reference",
            "title": title,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (
                f"https://doi.org/{doi}" if doi else None
            ),
            "pmid": pmid or None,
            "doi": doi or None,
        })
    return result


def catalogue_model(session: requests.Session, name: str) -> dict:
    response = session.get(DASHBOARD_URL, params={"search": name}, timeout=30)
    response.raise_for_status()
    matches = dashboard_payload(response.text).get("searchResults", {}).get("data", [])
    record = next((item for item in matches if normalize(item.get("name")) == normalize(name)), None)
    if record is None:
        candidates = [str(item.get("name")) for item in matches[:10]]
        raise LookupError(f"{name!r} was not found in the public catalogue; candidates: {candidates}")

    model_id = record["id"]
    detail = session.get(MODEL_URL.format(model_id), timeout=30)
    detail.raise_for_status()
    versions = detail.json().get("data", {}).get("versions") or []
    version = next((item for item in versions if item.get("default")), versions[0] if versions else {})
    references = public_references(version)
    return {
        "source": "Cell Collective",
        "cell_collective_id": model_id,
        "model_url": f"https://research.cellcollective.org/dashboard#module/{model_id}:1",
        "primary_reference": references[0] if references else None,
        "references": references,
        # Preserve the full public dashboard record without duplicating the
        # Boolean rules that CANA already distributes with the application.
        "catalogue": record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Metadata JSON destination")
    parser.add_argument("--pause", type=float, default=0.05, help="Seconds to wait between public API requests")
    args = parser.parse_args()

    names = sorted({model_name(model) for model in load_all_cell_collective_models()})
    session = requests.Session()
    session.headers.update({"User-Agent": "CANA-Streamlit-Demo metadata refresh"})
    models: dict[str, dict] = {}
    failures: list[str] = []

    for index, name in enumerate(names, start=1):
        try:
            models[name] = catalogue_model(session, name)
            print(f"[{index}/{len(names)}] {name}")
        except (requests.RequestException, ValueError, KeyError, TypeError, LookupError) as error:
            failures.append(f"{name}: {error}")
            print(f"[{index}/{len(names)}] FAILED {name}: {error}", file=sys.stderr)
        if args.pause > 0:
            time.sleep(args.pause)

    if failures:
        print("Metadata refresh was incomplete; refusing to overwrite the catalogue.", file=sys.stderr)
        return 1

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_count": len(models),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(models)} model records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Model catalogue, upload validation, and Cell Collective metadata access."""

import json
import os
import re
import tempfile
from html import escape, unescape
from typing import Any

import requests
import streamlit as st
from cana.boolean_network import BooleanNetwork as BN
from cana.datasets.bio import load_all_cell_collective_models
import cana.datasets.bio as bio

from constants import (
    CELL_COLLECTIVE_DASHBOARD_URL,
    CELL_COLLECTIVE_MODEL_URL,
    MAX_UPLOAD_EDGES,
    MAX_UPLOAD_NODE_INPUTS,
    MAX_UPLOAD_NODES,
    MODEL_METADATA_PATH,
)


def normalize_name(value: object) -> str:
    return str(value).strip().lower()


def get_bn_display_name(bn: BN, fallback: str = "Boolean Network") -> str:
    """Return a human-readable network name with a safe fallback."""
    name = str(getattr(bn, "name", "") or "").strip()
    return name or fallback


def _clone_if_possible(bn: BN) -> BN:
    try:
        return bn.copy() if hasattr(bn, "copy") else bn
    except Exception:
        return bn


def _load_bio_constant(model_name: str, dataset_obj: Any) -> BN:
    if isinstance(dataset_obj, BN):
        bn = _clone_if_possible(dataset_obj)
    elif isinstance(dataset_obj, str):
        try:
            bn = BN.from_file(dataset_obj, type="cnet")
        except Exception:
            bn = BN.from_file(dataset_obj)
    elif callable(dataset_obj):
        return _load_bio_constant(model_name, dataset_obj())
    else:
        raise ValueError(f"Could not load extra bio model: {model_name}")
    if not getattr(bn, "name", None):
        bn.name = model_name.replace("_", " ").title()
    return bn


@st.cache_resource(show_spinner=True)
def load_cell_collective_models() -> list[BN]:
    return list(load_all_cell_collective_models())


@st.cache_resource(show_spinner=True)
def load_extra_bio_models() -> tuple[dict[str, BN], dict[str, str]]:
    loaded: dict[str, BN] = {}
    failed: dict[str, str] = {}
    for constant in ("BREAST_CANCER", "BUDDING_YEAST", "DROSOPHILA", "LEUKEMIA", "MARQUESPITA", "THALIANA"):
        try:
            bn = _load_bio_constant(constant, getattr(bio, constant))
            name = get_bn_display_name(bn, constant.replace("_", " ").title())
            loaded[f"{name} ({constant})" if name in loaded else name] = bn
        except Exception as error:
            failed[constant] = str(error)
    return loaded, failed


@st.cache_resource(show_spinner=True, ttl=60 * 60, max_entries=8)
def load_uploaded_cnet_from_bytes(file_bytes: bytes, filename: str) -> BN:
    """Load an uploaded CNET file through CANA without retaining a temp file."""
    suffix = os.path.splitext(filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(file_bytes)
        temporary_path = temporary_file.name
    try:
        return BN.from_file(temporary_path, type="cnet")
    finally:
        try:
            os.remove(temporary_path)
        except OSError:
            pass


def validate_uploaded_network(bn: BN) -> Any:
    """Reject uploaded networks that exceed interactive rendering limits."""
    node_count = len(getattr(bn, "nodes", []) or [])
    if node_count == 0:
        raise ValueError("The uploaded model does not contain any nodes.")
    if node_count > MAX_UPLOAD_NODES:
        raise ValueError(f"This model contains {node_count:,} nodes; the interactive limit is {MAX_UPLOAD_NODES:,}.")
    graph = bn.structural_graph()
    if graph.number_of_edges() > MAX_UPLOAD_EDGES:
        raise ValueError(f"This model contains {graph.number_of_edges():,} edges; the interactive limit is {MAX_UPLOAD_EDGES:,}.")
    widest_input = max((getattr(node, "k", len(getattr(node, "inputs", []) or [])) for node in bn.nodes), default=0)
    if widest_input > MAX_UPLOAD_NODE_INPUTS:
        raise ValueError(f"A node has {widest_input} inputs; the interactive limit is {MAX_UPLOAD_NODE_INPUTS}.")
    return graph


def build_model_registry() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return all supported models with their provenance labels."""
    registry = {
        get_bn_display_name(model, "Unnamed Cell Collective Model"): {"bn": model, "source": "Cell Collective"}
        for model in load_cell_collective_models()
    }
    extras, failed = load_extra_bio_models()
    for name, model in extras.items():
        registry[f"{name} (extra bio)" if name in registry else name] = {"bn": model, "source": "CANA bio"}
    return registry, failed


def _references(version: dict[str, Any]) -> list[dict[str, str | None]]:
    references = version.get("referenceMap") or {}
    linked = version.get("modelReferenceMap") or {}
    records = []
    for item in sorted(linked.values(), key=lambda value: (value.get("position", float("inf")), value.get("referenceId", float("inf")))):
        reference = references.get(str(item.get("referenceId")))
        if not reference:
            continue
        clean = lambda value: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(str(value or "")))).strip()
        pmid, doi = str(reference.get("pmid") or "").strip(), str(reference.get("doi") or "").strip()
        records.append({
            "citation": clean(reference.get("text") or reference.get("shortCitation")) or "Reference",
            "title": clean(reference.get("title") or reference.get("articleTitle") or reference.get("publicationTitle")),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (f"https://doi.org/{doi}" if doi else None),
            "pmid": pmid or None,
            "doi": doi or None,
        })
    return records


@st.cache_data(show_spinner=False)
def load_local_model_metadata() -> dict[str, dict[str, Any]]:
    try:
        with open(MODEL_METADATA_PATH, encoding="utf-8") as metadata_file:
            records = json.load(metadata_file).get("models", {})
        return records if isinstance(records, dict) else {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def format_primary_citation(citation: str, paper_title: str = "") -> str:
    """Escape a paper citation and emphasize the identified title."""
    citation, title = str(citation or ""), str(paper_title or "").strip()
    start = citation.lower().find(title.lower()) if title else -1
    if start >= 0:
        end = start + len(title)
        return f'{escape(citation[:start])}<strong class="primary-paper-title">{escape(citation[start:end])}</strong>{escape(citation[end:])}'
    return escape(citation)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_live_cell_collective_source_info(model_name: str) -> dict[str, Any]:
    """Fetch public metadata only when the checked-in catalogue lacks a model."""
    try:
        response = requests.get(CELL_COLLECTIVE_DASHBOARD_URL, params={"search": model_name}, timeout=20)
        response.raise_for_status()
        marker = 'const data = {"published":'
        start = response.text.find(marker)
        if start < 0:
            raise ValueError("Cell Collective catalogue payload was not found.")
        results = json.JSONDecoder().raw_decode(response.text[start + len("const data = "):])[0].get("searchResults", {}).get("data", [])
        model = next((item for item in results if normalize_name(item.get("name")).rstrip(".") == normalize_name(model_name).rstrip(".")), None)
        if model is None:
            return {"error": "The selected model was not found in the public Cell Collective catalogue."}
        detail = requests.get(CELL_COLLECTIVE_MODEL_URL.format(model["id"]), timeout=20)
        detail.raise_for_status()
        versions = detail.json().get("data", {}).get("versions") or []
        version = next((item for item in versions if item.get("default")), versions[0] if versions else {})
        refs = _references(version)
        return {"model_url": f"https://research.cellcollective.org/dashboard#module/{model['id']}:1", "primary_reference": refs[0] if refs else None}
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        return {"error": str(error)}


def load_cell_collective_source_info(model_name: str) -> dict[str, Any]:
    return load_local_model_metadata().get(model_name) or load_live_cell_collective_source_info(model_name)

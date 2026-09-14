# Effective / Activity / Excess Canalization / Correlation Graph Explorer + Schemata Viewer

import os
import io
import base64
import hashlib
import json
import math
import re
import tempfile
from copy import copy
from html import escape, unescape

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle, RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.text import Text

import streamlit as st
import streamlit.components.v1 as components
import graphviz
import requests

from cana.boolean_network import BooleanNetwork as BN
from cana.datasets.bio import load_all_cell_collective_models
import cana.datasets.bio as bio
from cana.drawing.canalizing_map import draw_canalizing_map_graphviz


# -------------------- Page & style --------------------
st.set_page_config(page_title="Effective Graph Threshold Explorer", layout="wide")

RADIUS        = 5.0
CANVAS_INCH   = 5
NODE_PENWIDTH = 3.5
FONT_SIZE     = "10"
ARROWSIZE     = "0.7"
PENWIDTH_MAX  = 2.5

# Uploaded models are intentionally bounded because several CANA analyses scale
# exponentially with a node's input count.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_NODES = 500
MAX_UPLOAD_EDGES = 5_000
MAX_UPLOAD_NODE_INPUTS = 18
MAX_CORRELATION_INPUTS = 16
SESSION_CACHE_MAX_ENTRIES = 24

# Edge width range for structural metrics
MIN_WIDTH = 0.5
MAX_WIDTH = 4.0

DEFAULT_OUTLINE   = "#ff9896"
SPECIAL_OUTLINE   = "#ffdf0e"
ISOLATED_OUTLINE  = "#888888"

POS_EDGE_COLOR      = "black"
NEG_EDGE_COLOR      = "black"
NEG_EDGE_STYLE      = "dashed"
ZERO_EDGE_COLOR     = "red"
ZERO_EDGE_STYLE     = "dashed"

cmap = LinearSegmentedColormap.from_list('custom', ['white', '#d62728'])
cmap.set_under('#2ca02c')  # nodes with zero value → green

NETWORK_GRAPH_COMPONENT = components.declare_component(
    "network_graph_component",
    path=os.path.join(os.path.dirname(__file__), "network_graph_component"),
)
CASCI_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "casci_logo_200.jpg")


# -------------------- Helpers --------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()


def session_cached(namespace, key, factory):
    """Small per-session cache for mutable CANA-derived results."""
    cache = st.session_state.setdefault(namespace, {})
    if key not in cache:
        if len(cache) >= SESSION_CACHE_MAX_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = factory()
    return cache[key]


def model_cache_key(source_label, model_name, uploaded_bytes=None):
    if uploaded_bytes is not None:
        digest = hashlib.sha256(uploaded_bytes).hexdigest()
        return f"upload:{digest}"
    return f"builtin:{source_label}:{model_name}"


def get_bn_display_name(bn, fallback="Boolean Network"):
    name = getattr(bn, "name", None)
    if name is None:
        return fallback
    name = str(name).strip()
    return name if name else fallback


def clone_bn_if_possible(bn):
    try:
        if hasattr(bn, "copy"):
            return bn.copy()
    except Exception:
        pass
    return bn


def try_load_bio_constant(model_name, dataset_obj):
    if isinstance(dataset_obj, BN):
        bn = clone_bn_if_possible(dataset_obj)
        if not getattr(bn, "name", None):
            bn.name = model_name.replace("_", " ").title()
        return bn

    if isinstance(dataset_obj, str):
        load_attempts = [
            lambda: BN.from_file(dataset_obj, type='cnet'),
            lambda: BN.from_file(dataset_obj),
        ]
        for attempt in load_attempts:
            try:
                bn = attempt()
                if not getattr(bn, "name", None):
                    bn.name = model_name.replace("_", " ").title()
                return bn
            except Exception:
                pass

    if callable(dataset_obj):
        try:
            obj = dataset_obj()
            return try_load_bio_constant(model_name, obj)
        except Exception:
            pass

    raise ValueError(f"Could not load extra bio model: {model_name}")


@st.cache_resource(show_spinner=True)
def load_cell_collective_models():
    return list(load_all_cell_collective_models())


@st.cache_resource(show_spinner=True)
def load_extra_bio_models():
    extra_names = [
        "BREAST_CANCER",
        "BUDDING_YEAST",
        "DROSOPHILA",
        "LEUKEMIA",
        "MARQUESPITA",
        "THALIANA",
    ]

    loaded = {}
    failed = {}

    for const_name in extra_names:
        try:
            dataset_obj = getattr(bio, const_name)
            bn = try_load_bio_constant(const_name, dataset_obj)

            pretty_name = get_bn_display_name(
                bn,
                fallback=const_name.replace("_", " ").title()
            )

            if pretty_name in loaded:
                pretty_name = f"{pretty_name} ({const_name})"

            loaded[pretty_name] = bn
        except Exception as e:
            failed[const_name] = str(e)

    return loaded, failed


@st.cache_resource(show_spinner=True, ttl=60 * 60, max_entries=8)
def load_uploaded_cnet_from_bytes(file_bytes: bytes, filename: str):
    suffix = os.path.splitext(filename)[1] if filename else ".txt"
    if not suffix:
        suffix = ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        bn = BN.from_file(tmp_path, type='cnet')
        return bn
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def validate_uploaded_network(bn):
    """Reject uploaded networks that exceed safe interactive-analysis limits."""
    node_count = len(getattr(bn, "nodes", []) or [])
    if node_count == 0:
        raise ValueError("The uploaded model does not contain any nodes.")
    if node_count > MAX_UPLOAD_NODES:
        raise ValueError(
            f"This model contains {node_count:,} nodes; the interactive limit is "
            f"{MAX_UPLOAD_NODES:,}."
        )

    structural_graph = bn.structural_graph()
    edge_count = structural_graph.number_of_edges()
    if edge_count > MAX_UPLOAD_EDGES:
        raise ValueError(
            f"This model contains {edge_count:,} edges; the interactive limit is "
            f"{MAX_UPLOAD_EDGES:,}."
        )

    widest_node = max(
        getattr(node, "k", len(getattr(node, "inputs", []) or []))
        for node in bn.nodes
    ) if bn.nodes else 0
    if widest_node > MAX_UPLOAD_NODE_INPUTS:
        raise ValueError(
            f"A node has {widest_node} inputs; the interactive limit is "
            f"{MAX_UPLOAD_NODE_INPUTS}."
        )
    return structural_graph


def build_model_registry():
    registry = {}

    cc_models = load_cell_collective_models()
    for m in cc_models:
        display_name = get_bn_display_name(m, fallback="Unnamed Cell Collective Model")
        registry[display_name] = {
            "bn": m,
            "source": "Cell Collective"
        }

    extra_models, failed_extra = load_extra_bio_models()
    for display_name, bn in extra_models.items():
        if display_name in registry:
            display_name = f"{display_name} (extra bio)"
        registry[display_name] = {
            "bn": bn,
            "source": "CANA bio"
        }

    return registry, failed_extra


CELL_COLLECTIVE_DASHBOARD_URL = "https://research.cellcollective.org/research/dashboard/"
CELL_COLLECTIVE_MODEL_URL = "https://research.cellcollective.org/web/api/model/{}"
MODEL_METADATA_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.json")


def _cell_collective_dashboard_payload(page_text):
    """Extract the model catalogue embedded in the public dashboard page."""
    marker = 'const data = {"published":'
    start = page_text.find(marker)
    if start < 0:
        raise ValueError("Cell Collective catalogue payload was not found.")
    return json.JSONDecoder().raw_decode(page_text[start + len("const data = "):])[0]


def _model_references(version):
    """Return the ordered public references linked to a model version."""
    references = version.get("referenceMap") or {}
    model_references = version.get("modelReferenceMap") or {}
    ordered = sorted(
        model_references.values(),
        key=lambda item: (item.get("position", float("inf")), item.get("referenceId", float("inf"))),
    )
    collected = []
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
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (f"https://doi.org/{doi}" if doi else None)
        collected.append({
            "citation": citation or "Reference",
            "title": title,
            "url": url,
            "pmid": pmid or None,
            "doi": doi or None,
        })
    return collected


def _primary_model_reference(version):
    """Return the first/primary public paper linked to a model version."""
    references = _model_references(version)
    return references[0] if references else None


@st.cache_data(show_spinner=False)
def load_local_model_metadata():
    """Load the checked-in Cell Collective metadata catalogue, when available."""
    try:
        with open(MODEL_METADATA_PATH, encoding="utf-8") as metadata_file:
            payload = json.load(metadata_file)
        models = payload.get("models", {})
        return models if isinstance(models, dict) else {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def format_primary_citation(citation, paper_title=""):
    """Escape a citation and emphasize its paper title when it can be identified."""
    citation = str(citation or "")
    title = str(paper_title or "").strip()

    if title:
        title_start = citation.lower().find(title.lower())
        if title_start >= 0:
            title_end = title_start + len(title)
            return (
                f"{escape(citation[:title_start])}"
                f'<strong class="primary-paper-title">{escape(citation[title_start:title_end])}</strong>'
                f"{escape(citation[title_end:])}"
            )

    first_period = citation.find(".")
    second_period = citation.find(".", first_period + 1) if first_period >= 0 else -1
    if second_period > first_period + 1:
        title_start = first_period + 1
        title_end = second_period
        candidate = citation[title_start:title_end].strip()
        if len(candidate) >= 8:
            before = citation[:title_start]
            after = citation[title_end:]
            return (
                f"{escape(before)}"
                f'<strong class="primary-paper-title">{escape(candidate)}</strong>'
                f"{escape(after)}"
            )

    return escape(citation)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_live_cell_collective_source_info(model_name):
    """Fetch the public model page and linked paper when local metadata is absent."""
    try:
        response = requests.get(CELL_COLLECTIVE_DASHBOARD_URL, params={"search": model_name}, timeout=20)
        response.raise_for_status()
        matches = _cell_collective_dashboard_payload(response.text).get("searchResults", {}).get("data", [])
        model = next(
            (
                item for item in matches
                if _norm(item.get("name")).rstrip(".") == _norm(model_name).rstrip(".")
            ),
            None,
        )
        if model is None:
            return {"error": "The selected model was not found in the public Cell Collective catalogue."}

        detail_response = requests.get(CELL_COLLECTIVE_MODEL_URL.format(model["id"]), timeout=20)
        detail_response.raise_for_status()
        versions = detail_response.json().get("data", {}).get("versions") or []
        version = next((item for item in versions if item.get("default")), versions[0] if versions else {})
        return {
            "model_url": f"https://research.cellcollective.org/dashboard#module/{model['id']}:1",
            "primary_reference": _primary_model_reference(version),
        }
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        return {"error": str(error)}


def load_cell_collective_source_info(model_name):
    """Prefer checked-in metadata; retain the public API as a resilient fallback."""
    local_record = load_local_model_metadata().get(model_name)
    if isinstance(local_record, dict) and local_record.get("model_url"):
        return local_record
    return load_live_cell_collective_source_info(model_name)


def detect_special_nodes(bn, G):
    special_norm_labels = set()
    for node in bn.nodes:
        name = getattr(node, "name", f"node_{getattr(node, 'id', 'X')}")
        inputs = list(getattr(node, "inputs", []) or [])
        k = getattr(node, "k", len(inputs))

        def is_self(inp):
            try:
                nid = getattr(node, "id", None)
                iid = getattr(inp, "id", None)
                if (nid is not None) and (iid is not None):
                    return iid == nid
            except Exception:
                pass
            iname = getattr(inp, "name", str(inp))
            return _norm(iname) == _norm(name)

        if k == 0:
            special_norm_labels.add(_norm(name))
        elif k == 1 and len(inputs) == 1 and is_self(inputs[0]):
            special_norm_labels.add(_norm(name))

    g_labels = {n: G.nodes[n].get('label', str(n)) for n in G.nodes()}
    g_labels_norm = {n: _norm(lbl) for n, lbl in g_labels.items()}
    special_nodes = {n for n, lbln in g_labels_norm.items() if lbln in special_norm_labels}

    if not special_nodes:
        tmp = []
        for n in G.nodes():
            indeg = G.in_degree(n)
            has_self = G.has_edge(n, n)
            only_self = (indeg == 1 and has_self and all(p == n for p in G.predecessors(n)))
            if indeg == 0 or only_self:
                tmp.append(n)
        special_nodes = set(tmp)
    return special_nodes


def render_hoverable_network_graph(graph):
    """Render Graphviz SVG with click-to-focus neighbor highlighting."""
    svg = graph.pipe(format="svg").decode("utf-8")
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg, flags=re.IGNORECASE)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg, flags=re.IGNORECASE).strip()

    components.html(
        f"""
        <style>
          html, body {{ height: 100%; margin: 0; padding: 0; background: transparent; overflow: hidden; }}
          #network-graph {{ position: relative; width: 100%; height: 100%; }}
          #network-graph svg {{ display: block; width: 100%; height: 100%; margin: 0 auto; }}
          #network-graph g.node {{ cursor: pointer; transition: opacity 160ms ease; }}
          #network-graph g.edge {{ transition: opacity 160ms ease; }}
          #network-graph g.node.dimmed, #network-graph g.edge.dimmed {{ opacity: 0.13; }}
          #network-graph g.node.focused ellipse {{ stroke: #0f172a !important; stroke-width: 6px !important; }}
          #network-graph g.node.neighbor ellipse {{ stroke: #2563eb !important; stroke-width: 5px !important; }}
          #network-graph g.edge.connected path {{ stroke: #2563eb !important; stroke-width: 3px !important; }}
          #network-graph g.edge.connected polygon {{ fill: #2563eb !important; stroke: #2563eb !important; }}
          #network-graph g.edge.incoming path {{ stroke: #be123c !important; }}
          #network-graph g.edge.incoming polygon {{ fill: #be123c !important; stroke: #be123c !important; }}
          #network-graph g.edge.outgoing path {{ stroke: #0891b2 !important; }}
          #network-graph g.edge.outgoing polygon {{ fill: #0891b2 !important; stroke: #0891b2 !important; }}
          #network-graph g.edge.incoming.outgoing path {{ stroke: #be123c !important; }}
          #network-graph g.edge.incoming.outgoing polygon {{ fill: #be123c !important; stroke: #be123c !important; }}
        </style>
        <div id="network-graph">{svg}</div>
        <script>
          const nodes = [...document.querySelectorAll("#network-graph g.node")];
          const edges = [...document.querySelectorAll("#network-graph g.edge")];
          let focusedNodeId = null;

          function graphId(element) {{
            return element.dataset.graphId || "";
          }}

          [...nodes, ...edges].forEach((element) => {{
            const title = element.querySelector("title");
            element.dataset.graphId = title ? title.textContent.trim() : "";
          }});
          document.querySelectorAll("#network-graph title").forEach((title) => title.remove());

          function edgeEndpoints(edge) {{
            const parts = graphId(edge).split("->");
            return parts.length === 2 ? parts.map((part) => part.trim()) : [];
          }}

          function clearFocus() {{
            focusedNodeId = null;
            nodes.forEach((node) => node.classList.remove("dimmed", "focused", "neighbor"));
            edges.forEach((edge) => edge.classList.remove(
              "dimmed", "connected", "incoming", "outgoing"
            ));
          }}

          function focusNode(node) {{
            const nodeId = graphId(node);
            if (!nodeId || focusedNodeId === nodeId) {{
              clearFocus();
              return;
            }}

            focusedNodeId = nodeId;
            const relatedIds = new Set([nodeId]);
            const relatedEdges = new Set();
            edges.forEach((edge) => {{
              const [source, target] = edgeEndpoints(edge);
              if (source === nodeId || target === nodeId) {{
                relatedIds.add(source);
                relatedIds.add(target);
                relatedEdges.add(edge);
              }}
            }});

            nodes.forEach((item) => {{
              const isFocused = graphId(item) === nodeId;
              item.classList.toggle("focused", isFocused);
              item.classList.toggle("neighbor", !isFocused && relatedIds.has(graphId(item)));
              item.classList.toggle("dimmed", !relatedIds.has(graphId(item)));
            }});
            edges.forEach((edge) => {{
              const [source, target] = edgeEndpoints(edge);
              const isConnected = relatedEdges.has(edge);
              edge.classList.toggle("connected", isConnected);
              edge.classList.toggle("dimmed", !isConnected);
              edge.classList.toggle("incoming", target === nodeId);
              edge.classList.toggle("outgoing", source === nodeId);
            }});
          }}

          nodes.forEach((node) => {{
            node.addEventListener("click", (event) => {{
              event.stopPropagation();
              focusNode(node);
            }});
          }});
          document.querySelector("#network-graph svg").addEventListener("click", clearFocus);
        </script>
        """,
        height=650,
        scrolling=False,
    )


def render_clickable_network_graph(graph, focused_node_id, context_id):
    """Render the network component and return the most recent click event."""
    svg = graphviz_svg_from_source(graph.source, graph.engine)
    return NETWORK_GRAPH_COMPONENT(
        svg=svg,
        focused_node_id=str(focused_node_id or ""),
        context_id=str(context_id or ""),
        key="network-graph-component",
        default={},
    )


@st.cache_data(show_spinner=False, max_entries=32)
def graphviz_svg_from_source(source, engine="dot"):
    """Render and cache deterministic Graphviz source as embeddable SVG."""
    svg = graphviz.Source(source, engine=engine).pipe(format="svg").decode("utf-8")
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg, flags=re.IGNORECASE)
    return re.sub(r"<!DOCTYPE[^>]*>", "", svg, flags=re.IGNORECASE).strip()


def circular_positions(G, radius=RADIUS):
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return [], {}

    label = lambda n: G.nodes[n].get('label', str(n))
    sorted_nodes = sorted(nodes, key=label)
    pos = {
        n: (radius * np.cos(2 * np.pi * i / N), radius * np.sin(2 * np.pi * i / N))
        for i, n in enumerate(sorted_nodes)
    }
    return sorted_nodes, pos


def threshold_graph(EG0, thr):
    EGf = EG0.copy()
    EGf.remove_edges_from([
        (u, v) for u, v, d in EGf.edges(data=True)
        if float(d.get('weight', 0.0)) <= thr
    ])
    return EGf


def colorbar_figure(max_val, label_text):
    cmap_local = LinearSegmentedColormap.from_list('custom', ['white', '#d62728'])
    cmap_local.set_under('#2ca02c')

    norm = mpl.colors.Normalize(vmin=1e-16, vmax=max_val if max_val > 0 else 1)

    fig = plt.figure(figsize=(0.8, 1.6), dpi=200)
    ax = fig.add_axes([0.03, 0.05, 0.15, 0.6])

    tick_max = int(np.ceil(max_val))
    ticks = list(range(0, tick_max + 1, max(1, tick_max // 4 or 1)))
    boundaries = np.linspace(-1, max_val, 25).tolist()

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap_local,
        norm=norm,
        boundaries=boundaries,
        ticks=ticks,
        spacing='uniform',
        orientation='vertical',
        extend='min',
        format='%.0f',
    )

    cb.set_label(label_text, fontsize=4, labelpad=8)
    cb.ax.tick_params(labelsize=9)

    return fig


def isolated_nodes_after_threshold(G, edge_values=None, thr=None, use_abs=False):
    H = G.copy()
    if edge_values is not None and thr is not None:
        to_remove = []
        for u, v in H.edges():
            key = (u, v)
            if key in edge_values:
                val = edge_values[key]
                test_val = abs(val) if use_abs else val
                if test_val <= thr:
                    to_remove.append((u, v))
        H.remove_edges_from(to_remove)

    return {
        n for n in H.nodes()
        if H.in_degree(n) == 0 and H.out_degree(n) == 0
    }


def node_values_from_thresholded_effective(EGf, degree_mode="Out-degree"):
    if degree_mode == "In-degree":
        vals = {n: EGf.in_degree(n, weight='weight') for n in EGf.nodes()}
    else:
        vals = {n: EGf.out_degree(n, weight='weight') for n in EGf.nodes()}
    for n in EGf.nodes():
        vals.setdefault(n, 0.0)
    return vals


def node_values_from_thresholded_structural(SG, edge_values, thr, degree_mode="Out-degree", use_abs=False):
    node_vals = {n: 0.0 for n in SG.nodes()}
    for (u, v), val in edge_values.items():
        comp_val = abs(val) if use_abs else val
        if comp_val > thr:
            add_val = abs(val) if use_abs else max(0.0, val)
            if degree_mode == "In-degree":
                node_vals[v] += add_val
            else:
                node_vals[u] += add_val
    return node_vals


# ---------- Curved zero-edge helpers ----------
_COMPASS = ["e", "ne", "n", "nw", "w", "sw", "s", "se"]

def _angle_to_compass(dx, dy):
    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    idx = int(round(angle / 45.0)) % 8
    return _COMPASS[idx]

def _shift_compass(port, step):
    idx = _COMPASS.index(port)
    return _COMPASS[(idx + step) % 8]

def curved_zero_edge_ports(u, v, positions, should_curve=False):
    """
    Keep normal edges on the default path.
    Only curve a zero edge when it overlaps with an opposite-direction edge.
    """
    if not should_curve:
        return {}

    if u not in positions or v not in positions:
        return {}

    x1, y1 = positions[u]
    x2, y2 = positions[v]
    dx = x2 - x1
    dy = y2 - y1

    if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
        return {}

    tail_base = _angle_to_compass(dx, dy)
    head_base = _angle_to_compass(-dx, -dy)

    # shift both ends to the same side for a mild curved/offset route
    tail_port = _shift_compass(tail_base, 1)
    head_port = _shift_compass(head_base, 1)

    return {
        "tailport": tail_port,
        "headport": head_port,
    }


def has_overlapping_opposite_edge(graph_obj, u, v):
    """Return True when the reverse edge also exists, so the two edges would overlap on the same centerline."""
    try:
        return graph_obj.has_edge(v, u)
    except Exception:
        return False


# ---------- Effective-graph visualization ----------
def build_graphviz_effective(
    EG0,
    node_values,
    special_nodes_set,
    positions,
    node_width_in,
    isolated_nodes=None,
):
    if isolated_nodes is None:
        isolated_nodes = set()

    for n in EG0.nodes():
        node_values.setdefault(n, 0.0)

    max_node_val = max(node_values.values()) if node_values else 1.0
    norm_mpl = mpl.colors.Normalize(vmin=1e-16, vmax=max_node_val if max_node_val > 0 else 1)

    g = graphviz.Digraph(engine='neato')
    g.attr(
        'graph',
        size=f'{CANVAS_INCH},{CANVAS_INCH}!',
        ratio='1',
        margin='0.2',
        pad='0.1',
        splines='true',
    )
    g.attr(
        'node',
        pin='true',
        shape='circle',
        fixedsize='true',
        width=f"{node_width_in:.2f}",
        style='filled',
        fontname='Helvetica',
        fontsize=FONT_SIZE,
        penwidth=str(NODE_PENWIDTH)
    )
    g.attr('edge', arrowhead='normal', arrowsize=ARROWSIZE, color='black')

    label = lambda n: EG0.nodes[n].get('label', str(n))
    for n, (x, y) in positions.items():
        val = node_values.get(n, 0.0)
        if val == 0:
            fill = '#2ca02c'
        else:
            fill = mpl.colors.rgb2hex(cmap(norm_mpl(val)))

        if n in isolated_nodes:
            outline = ISOLATED_OUTLINE
        elif n in special_nodes_set:
            outline = SPECIAL_OUTLINE
        else:
            outline = DEFAULT_OUTLINE

        g.node(
            str(n),
            label(n),
            pos=f"{x:.3f},{y:.3f}!",
            color=outline,
            fillcolor=fill,
        )

    return g, max_node_val


def add_edges_effective(g, EG0, thr, positions):
    normal_edges = []
    zero_edges = []

    for u, v, d in EG0.edges(data=True):
        w = float(d.get('weight', 0.0))

        if np.isclose(thr, 0.0) and np.isclose(w, 0.0):
            zero_edges.append((u, v))
            continue

        if w <= thr:
            continue

        normal_edges.append((u, v, w))

    for u, v, w in normal_edges:
        pen = max(0.5, min(PENWIDTH_MAX, PENWIDTH_MAX * w)) if w > 0 else 1.5
        g.edge(str(u), str(v), penwidth=f"{pen:.2f}", color="black")

    for u, v in zero_edges:
        port_kwargs = curved_zero_edge_ports(
            u, v, positions, should_curve=has_overlapping_opposite_edge(EG0, u, v)
        )
        g.edge(
            str(u), str(v),
            penwidth="3.5",
            color=ZERO_EDGE_COLOR,
            style=ZERO_EDGE_STYLE,
            **port_kwargs
        )


# ---------- Structural metrics ----------
def metric_to_width(val, vmin, vmax, wmin=MIN_WIDTH, wmax=MAX_WIDTH):
    if vmax > vmin:
        t = (val - vmin) / (vmax - vmin)
    else:
        t = 0.5
    return wmin + t * (wmax - wmin)


def _graph_node_for_reference(reference, graph, nodes_by_normalized_label):
    """Resolve a CANA node/input reference to the matching graph node ID."""
    candidates = [reference, getattr(reference, "id", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if candidate in graph:
                return candidate
        except TypeError:
            pass

    names = [getattr(reference, "name", None), str(reference)]
    for name in names:
        if name is not None:
            match = nodes_by_normalized_label.get(_norm(name))
            if match is not None:
                return match
    return None


def _ordered_cana_edges(node, graph):
    """Return graph edges in CANA's declared input order."""
    nodes_by_label = {
        _norm(graph.nodes[node_id].get("label", node_id)): node_id
        for node_id in graph.nodes()
    }
    target = _graph_node_for_reference(node, graph, nodes_by_label)
    if target is None:
        return []

    ordered_edges = []
    for input_reference in list(getattr(node, "inputs", []) or []):
        source = _graph_node_for_reference(input_reference, graph, nodes_by_label)
        ordered_edges.append((source, target) if source is not None and graph.has_edge(source, target) else None)
    return ordered_edges


def compute_structural_metrics(bn, include_excess=True):
    SG = bn.structural_graph()
    if SG.number_of_nodes() == 0:
        return SG, {}, (0.0, 1.0), {}, (0.0, 1.0)

    edge_activity = {}
    edge_activity_vals = []
    edge_excess = {}
    edge_excess_vals = []

    for node in bn.nodes:
        try:
            act_list = node.activities()
            eff_list = node.edge_effectiveness() if include_excess else None
        except Exception:
            continue

        if act_list is None:
            continue

        act_list = list(act_list)
        eff_list = list(eff_list) if eff_list is not None else None

        if len(act_list) == 0:
            continue

        ordered_edges = _ordered_cana_edges(node, SG)
        L = min(len(ordered_edges), len(act_list))
        if L == 0:
            continue

        for i in range(L):
            edge = ordered_edges[i]
            if edge is None:
                continue
            act = act_list[i]
            if not np.isfinite(act):
                continue

            u, v = edge
            act_val = float(act)
            edge_activity[(u, v)] = act_val
            edge_activity_vals.append(act_val)

            if eff_list is not None and i < len(eff_list):
                eff = eff_list[i]
                if np.isfinite(eff):
                    exc = float(eff - act)
                    edge_excess[(u, v)] = exc
                    edge_excess_vals.append(exc)

    if not edge_activity_vals:
        act_min, act_max = 0.0, 1.0
    else:
        arr = np.array(edge_activity_vals)
        act_min, act_max = arr.min(), arr.max()

    if not edge_excess_vals:
        ex_min, ex_max = 0.0, 1.0
    else:
        arr = np.array(edge_excess_vals)
        ex_min, ex_max = arr.min(), arr.max()

    return SG, edge_activity, (act_min, act_max), edge_excess, (ex_min, ex_max)


# ---------- Correlation metric ----------
def safe_binary_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0

    c = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)


def compute_correlation_metrics(bn):
    SG = bn.structural_graph()
    if SG.number_of_nodes() == 0:
        return SG, {}, (0.0, 1.0)

    edge_corr = {}
    corr_abs_vals = []

    for node in bn.nodes:
        inputs = list(getattr(node, "inputs", []) or [])
        k = getattr(node, "k", len(inputs))

        if k <= 0 or len(inputs) == 0:
            continue
        if k > MAX_CORRELATION_INPUTS:
            name = getattr(node, "name", f"node_{getattr(node, 'id', 'X')}")
            raise ValueError(
                f"Correlation is limited to {MAX_CORRELATION_INPUTS} inputs per node; "
                f"{name} has {k}."
            )

        outputs = getattr(node, "outputs", None)
        if outputs is None:
            continue

        outputs = np.asarray(list(outputs), dtype=float)
        if outputs.size != 2 ** k:
            continue

        ordered_edges = _ordered_cana_edges(node, SG)
        row_ids = np.arange(outputs.size, dtype=np.uint64)
        L = min(len(ordered_edges), k)
        for i in range(L):
            edge = ordered_edges[i]
            if edge is None:
                continue
            u, v = edge
            col = ((row_ids >> np.uint64(k - i - 1)) & np.uint64(1)).astype(float)
            corr = safe_binary_corr(col, outputs)
            edge_corr[(u, v)] = corr
            corr_abs_vals.append(abs(corr))

    if not corr_abs_vals:
        cmin, cmax = 0.0, 1.0
    else:
        arr = np.array(corr_abs_vals)
        cmin, cmax = float(arr.min()), float(arr.max())

    return SG, edge_corr, (cmin, cmax)


def build_graphviz_structural(SG, node_values, special_nodes_set, positions, node_width_in, isolated_nodes=None):
    if isolated_nodes is None:
        isolated_nodes = set()

    for n in SG.nodes():
        node_values.setdefault(n, 0.0)

    max_val = max(node_values.values()) if node_values else 1.0
    norm_mpl = mpl.colors.Normalize(vmin=1e-16, vmax=max_val if max_val > 0 else 1)

    g = graphviz.Digraph(engine='neato')
    g.attr(
        'graph',
        size=f'{CANVAS_INCH},{CANVAS_INCH}!',
        ratio='1',
        margin='0.2',
        pad='0.1',
        splines='true',
    )
    g.attr(
        'node',
        pin='true',
        shape='circle',
        fixedsize='true',
        width=f"{node_width_in:.2f}",
        style='filled',
        fontname='Helvetica',
        fontsize=FONT_SIZE,
        penwidth=str(NODE_PENWIDTH)
    )
    g.attr('edge', arrowhead='normal', arrowsize=ARROWSIZE, color='black')

    label = lambda n: SG.nodes[n].get('label', str(n))
    for n, (x, y) in positions.items():
        val = node_values.get(n, 0.0)
        if val <= 0:
            fill = '#2ca02c'
        else:
            fill = mpl.colors.rgb2hex(cmap(norm_mpl(val)))

        if n in isolated_nodes:
            outline = ISOLATED_OUTLINE
        elif n in special_nodes_set:
            outline = SPECIAL_OUTLINE
        else:
            outline = DEFAULT_OUTLINE

        g.node(
            str(n),
            label(n),
            pos=f"{x:.3f},{y:.3f}!",
            color=outline,
            fillcolor=fill,
        )
    return g, max_val


def add_edges_structural(
    g,
    SG,
    edge_values,
    vmin,
    vmax,
    positions,
    thr=None,
    show_zero_dashed_at_zero_threshold=False,
    signed_color=False,
    threshold_on_abs=False,
    negative_dashed=False,
):
    normal_edges = []
    zero_edges = []

    for u, v, d in SG.edges(data=True):
        key = (u, v)

        if key not in edge_values:
            continue

        val = float(edge_values[key])
        comp_val = abs(val) if threshold_on_abs else val

        if (
            show_zero_dashed_at_zero_threshold
            and thr is not None
            and np.isclose(thr, 0.0)
            and np.isclose(val, 0.0)
        ):
            zero_edges.append((u, v))
            continue

        if thr is not None and comp_val <= thr:
            continue

        normal_edges.append(("metric", u, v, val))

    for item in normal_edges:
        if item[0] == "default":
            _, u, v = item
            g.edge(str(u), str(v), penwidth="1.0", color="#2ca02c")
        else:
            _, u, v, val = item

            width_val = abs(val) if threshold_on_abs else val
            width = metric_to_width(width_val, vmin, vmax)

            color = "black"
            style = "solid"

            if signed_color:
                if val < 0:
                    color = NEG_EDGE_COLOR
                    if negative_dashed:
                        style = NEG_EDGE_STYLE
                else:
                    color = POS_EDGE_COLOR

            g.edge(
                str(u), str(v),
                penwidth=f"{width:.2f}",
                color=color,
                style=style
            )

    for u, v in zero_edges:
        port_kwargs = curved_zero_edge_ports(
            u, v, positions, should_curve=has_overlapping_opposite_edge(SG, u, v)
        )
        g.edge(
            str(u), str(v),
            penwidth="3.5",
            color=ZERO_EDGE_COLOR,
            style=ZERO_EDGE_STYLE,
            **port_kwargs
        )


# ---------- Legend ----------
def _legend_line(color="black", style="solid", width=3):
    dash_style = "solid" if style == "solid" else "dashed"
    return (
        f"<span style='display:inline-block; width:36px; vertical-align:middle; "
        f"border-top:{width}px {dash_style} {color}; margin-right:8px;'></span>"
    )


def _legend_node(fill="#ffffff", outline="#000000"):
    return (
        f"<span style='display:inline-block; width:16px; height:16px; border-radius:50%; "
        f"background:{fill}; border:3px solid {outline}; margin-right:8px; vertical-align:middle;'></span>"
    )


def render_graph_legend(metric, threshold_value, degree_mode, scale_max):
    node_items = [
        (_legend_node(fill="#f7f7f7", outline=DEFAULT_OUTLINE), "regular node"),
        (_legend_node(fill="#f7f7f7", outline=SPECIAL_OUTLINE), "input node"),
        (_legend_node(fill="#f7f7f7", outline=ISOLATED_OUTLINE), "isolated after thresholding"),
    ]

    if metric == "Edge effectiveness":
        edge_items = [
            (_legend_line(color="black", style="solid", width=3), "thicker = larger effectiveness"),
            (_legend_line(color="red", style="dashed", width=3), "weight = 0"),
        ]
    elif metric == "Activity":
        edge_items = [
            (_legend_line(color="black", style="solid", width=3), "thicker = larger activity"),
            (_legend_line(color="red", style="dashed", width=3), "activity = 0"),
        ]
    elif metric == "Excess canalization":
        edge_items = [
            (_legend_line(color="black", style="solid", width=3), "thicker = larger excess canalization"),
        ]
    else:
        edge_items = [
            (_legend_line(color="black", style="solid", width=3), "positive correlation; thicker = larger |correlation|"),
            (_legend_line(color="black", style="dashed", width=3), "negative correlation"),
            (_legend_line(color="red", style="dashed", width=3), "correlation = 0"),
        ]

    row_style = "margin:6px 0; padding:6px 8px; border-radius:10px; background:rgba(248,250,252,0.95); border:1px solid rgba(148,163,184,0.14);"
    node_html = "".join([
        f"<div style='{row_style}'>{icon}<span>{label}</span></div>"
        for icon, label in node_items
    ])
    edge_html = "".join([
        f"<div style='{row_style}'>{icon}<span>{label}</span></div>"
        for icon, label in edge_items
    ])

    scale_title = f"Node color scale ({degree_mode.lower()})"
    scale_min = 0.0
    scale_max = float(scale_max) if float(scale_max) > 0 else 0.0

    gradient_bar = (
        "linear-gradient(to right, "
        "#2ca02c 0%, "
        "#f7f7f7 18%, "
        "#fddbc7 45%, "
        "#fca082 70%, "
        "#f1695c 85%, "
        "#d62728 100%)"
    )

    scale_html = f"""
    <div style="margin-top:6px; padding:8px 10px; border-radius:12px; background:rgba(248,250,252,0.95); border:1px solid rgba(148,163,184,0.14);">
        <div style="
            width:100%;
            height:16px;
            border:1px solid #999;
            border-radius:8px;
            background:{gradient_bar};
        "></div>
        <div style="display:flex; justify-content:space-between; font-size:0.9em; margin-top:4px;">
            <span>0.00</span>
            <span>{scale_max:.2f}</span>
        </div>
        <div style="font-size:0.9em; color:#444; margin-top:4px;">
            Green = zero, darker red = larger {degree_mode.lower()} value
        </div>
    </div>
    """

    html = f"""
    <div style="font-weight:700; font-size:0.88rem; letter-spacing:0.02em; margin-bottom:6px; color:#0f172a;">Nodes</div>
    {node_html}
    <div style="font-weight:700; font-size:0.88rem; letter-spacing:0.02em; margin:12px 0 6px 0; color:#0f172a;">Edges</div>
    {edge_html}
    <div style="font-weight:700; font-size:0.88rem; letter-spacing:0.02em; margin:12px 0 6px 0; color:#0f172a;">{scale_title}</div>
    {scale_html}
    """
    st.markdown(html, unsafe_allow_html=True)


def clean_svg_for_panel(svg_text):
    svg_text = re.sub(r"<\?xml[^>]*\?>", "", svg_text, flags=re.IGNORECASE)
    svg_text = re.sub(r"<!DOCTYPE[^>]*>", "", svg_text, flags=re.IGNORECASE)
    svg_text = re.sub(r"</?div[^>]*>", "", svg_text, flags=re.IGNORECASE).strip()

    # Remove fixed SVG size attributes so the browser can scale tall figures more
    # reliably inside the square panel without clipping.
    svg_text = re.sub(r'\swidth="[^"]*"', '', svg_text, count=1)
    svg_text = re.sub(r'\sheight="[^"]*"', '', svg_text, count=1)

    if 'preserveAspectRatio=' in svg_text:
        svg_text = re.sub(
            r'preserveAspectRatio="[^"]*"',
            'preserveAspectRatio="xMidYMid meet"',
            svg_text,
            count=1
        )
    else:
        svg_text = re.sub(
            r'<svg\b',
            '<svg preserveAspectRatio="xMidYMid meet"',
            svg_text,
            count=1
        )

    if 'class="canalization-map-svg"' not in svg_text:
        svg_text = re.sub(
            r'<svg\b',
            '<svg class="canalization-map-svg"',
            svg_text,
            count=1
        )

    return svg_text


def format_node_parameter(metric_fn):
    """Return a consistently formatted node metric, or an em dash when unavailable."""
    try:
        value = float(metric_fn())
        return f"{value:.2f}" if np.isfinite(value) else "—"
    except Exception:
        return "—"


# ---------- Schemata plotting ----------
def _safe_compute_schemata(node):
    try:
        node._check_compute_canalization_variables(prime_implicants=True)
    except Exception:
        pass

    try:
        node._check_compute_canalization_variables(two_symbols=True)
    except Exception:
        pass

    if (getattr(node, "_prime_implicants", None) is None) or (getattr(node, "_two_symbols", None) is None):
        try:
            if hasattr(node, "schemata"):
                node.schemata()
        except Exception:
            pass


def plot_schemata(n):
    _safe_compute_schemata(n)

    k = n.k if n.k >= 1 else 1
    inputs = n.inputs if not n.constant else [n.name]
    inputlabels = [n.network.get_node_name(i)[0] if n.network is not None else i for i in inputs]

    pi_dict = getattr(n, "_prime_implicants", {}) or {}
    pi0s = pi_dict.get('0', []) or []
    pi1s = pi_dict.get('1', []) or []

    two_symbols = getattr(n, "_two_symbols", None)
    if two_symbols is None:
        ts0s, ts1s = [], []
    else:
        ts0s = two_symbols[0] if len(two_symbols) > 0 and two_symbols[0] is not None else []
        ts1s = two_symbols[1] if len(two_symbols) > 1 and two_symbols[1] is not None else []

    n_pi = sum(len(pis) for pis in [pi0s, pi1s])
    n_ts = sum(len(tss) for tss in [ts0s, ts1s])

    cwidth = 60.0
    cxspace = 0
    cyspace = 6
    border = 1
    sepcxspace = 21
    sepcyspace = 15
    dpi = 150.0

    top, right, bottom, left, hs = 160, 25, 25, 60, 60

    ax1width = (k * (cwidth + cxspace)) + sepcxspace + cwidth
    ax2width = (k * (cwidth + cxspace)) + sepcxspace + cwidth

    n_pi_eff = max(n_pi, 1)
    n_ts_eff = max(n_ts, 1)

    ax1height = n_pi_eff * (cwidth + cyspace) + sepcyspace - cyspace
    ax2height = n_ts_eff * (cwidth + cyspace) + sepcyspace - cyspace

    fwidth = left + ax1width + hs + ax2width + right
    fheight = bottom + max(ax1height, ax2height) + top

    _ax1w = ax1width / fwidth
    _ax2w = ax2width / fwidth
    _ax1h = ax1height / fheight
    _ax2h = ax2height / fheight
    _bottom = bottom / fheight
    _left = left / fwidth
    _hs = hs / fwidth

    fig = plt.figure(figsize=(fwidth / dpi, fheight / dpi), facecolor='w', dpi=dpi)
    ax1 = fig.add_axes((_left, _bottom, _ax1w, _ax1h), aspect=1, label='PI')
    ax2 = fig.add_axes((_left + _ax1w + _hs, _bottom, _ax2w, _ax2h), aspect=1, label='TS')

    yticks = []
    patches = []
    x, y = 0.0, 0.0
    xticks_pi = []

    for out, pis in zip([1, 0], [pi1s, pi0s]):
        for pi in pis:
            x = 0.0
            xticks_pi = []
            for input_val in pi:
                if input_val == '0':
                    facecolor = 'white'
                    textcolor = 'black'
                elif input_val == '1':
                    facecolor = 'black'
                    textcolor = 'white'
                elif input_val in ['#', '2']:
                    facecolor = '#cccccc'
                    textcolor = 'black'
                else:
                    facecolor = 'white'
                    textcolor = 'black'

                text = f'{input_val}' if input_val != '2' else '#'
                ax1.add_artist(Text(
                    x + cwidth / 2, y + cwidth / 10 * 4,
                    text=text, color=textcolor, va='center', ha='center',
                    fontsize=14, family='serif'
                ))
                r = Rectangle((x, y), width=cwidth, height=cwidth, facecolor=facecolor, edgecolor='black')
                patches.append(r)
                xticks_pi.append(x + cwidth / 2)
                x += cwidth + cxspace

            x += sepcxspace
            r = Rectangle(
                (x, y), width=cwidth, height=cwidth,
                facecolor='black' if out == 1 else 'white',
                edgecolor='black'
            )
            ax1.add_artist(Text(
                x - (sepcxspace / 2) - (cxspace / 2), y + cwidth / 10 * 4,
                text=':', color='black', va='center', ha='center',
                fontsize=14, weight='bold', family='serif'
            ))
            ax1.add_artist(Text(
                x + (cwidth / 2), y + cwidth / 10 * 4,
                text=str(out),
                color='white' if out == 1 else 'black',
                va='center', ha='center',
                fontsize=14, family='serif'
            ))
            patches.append(r)
            xticks_pi.append(x + cwidth / 2)
            yticks.append(y + cwidth / 2)
            y += cwidth + cyspace
        y += sepcyspace

    if patches:
        ax1.add_collection(PatchCollection(patches, match_original=True))

    ax1.set_yticks(yticks)
    ax1.set_yticklabels([r"$f^{'}_{%d}$" % (i + 1) for i in range(n_pi)[::-1]], fontsize=14)

    if xticks_pi:
        ax1.set_xticks(xticks_pi)
        ax1.set_xticklabels(inputlabels + [f'{n.name}'], rotation=90, fontsize=14)
    else:
        ax1.set_xticks([])

    ax1.xaxis.tick_top()
    ax1.tick_params(which='major', pad=7)
    for tic in ax1.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax1.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax1.set_xlim(-border, ax1width + border)
    ax1.set_ylim(-border, ax1height + border)

    x, y = 0.0, 0.0
    yticks = []
    boxes, symbols = [], []
    xticks_ts = []

    tssymbols = [
        Circle((0, 0), radius=5, facecolor='white', edgecolor='black'),
        RegularPolygon((0, 0), numVertices=3, radius=5, orientation=0, facecolor='white', edgecolor='black'),
    ]

    for out, tss in zip([1, 0], [ts1s, ts0s]):
        for ts, pss, sss in tss:
            x = 0.0
            xticks_ts = []
            for i, input_val in enumerate(ts):
                if input_val == '0':
                    facecolor = 'white'
                    textcolor = 'black'
                elif input_val == '1':
                    facecolor = 'black'
                    textcolor = 'white'
                elif input_val == '2':
                    facecolor = '#cccccc'
                    textcolor = 'black'
                else:
                    facecolor = 'white'
                    textcolor = 'black'

                if len(pss):
                    iinpss = [j for j, ps in enumerate(pss) if i in ps]
                    xpos = np.linspace(x, x + cwidth, len(iinpss) + 2)
                    for z, j in enumerate(iinpss, start=1):
                        if j >= len(tssymbols):
                            continue
                        s = copy(tssymbols[j])
                        s.set_facecolor('none')
                        if hasattr(s, "xy"):
                            s.xy = (xpos[z], y + cwidth * 0.8)
                        if hasattr(s, "center"):
                            s.center = (xpos[z], y + cwidth * 0.8)
                        s.set_zorder(10)
                        s.set_edgecolor('#a6a6a6' if input_val == '1' else 'black')
                        symbols.append(s)
                        ax2.add_patch(s)

                text = f'{input_val}' if input_val != '2' else '#'
                ax2.add_artist(Text(
                    x + cwidth / 2, y + cwidth / 10 * 4,
                    text=text, color=textcolor, va='center', ha='center',
                    fontsize=14, family='serif'
                ))
                r = Rectangle((x, y), width=cwidth, height=cwidth, facecolor=facecolor, edgecolor='#4c4c4c', zorder=2)
                boxes.append(r)
                xticks_ts.append(x + cwidth / 2)
                x += cwidth + cxspace

            x += sepcxspace
            r = Rectangle(
                (x, y), width=cwidth, height=cwidth,
                facecolor='black' if out == 1 else 'white',
                edgecolor='#4c4c4c'
            )
            ax2.add_artist(Text(
                x - (sepcxspace / 2) - (cxspace / 2), y + cwidth / 2,
                text=':', color='black', va='center', ha='center',
                fontsize=14, weight='bold', family='serif'
            ))
            ax2.add_artist(Text(
                x + (cwidth / 2), y + cwidth / 10 * 4,
                text=str(out),
                color='white' if out == 1 else 'black',
                va='center', ha='center',
                fontsize=14, family='serif'
            ))
            boxes.append(r)
            xticks_ts.append(x + cwidth / 2)
            yticks.append(y + cwidth / 2)
            y += cwidth + cyspace
        y += sepcyspace

    if boxes:
        ax2.add_collection(PatchCollection(boxes, match_original=True))
    if symbols:
        ax2.add_collection(PatchCollection(symbols, match_original=True))

    ax2.set_yticks(yticks)
    ax2.set_yticklabels([r"$f^{''}_{%d}$" % (i + 1) for i in range(n_ts)[::-1]], fontsize=14)

    if xticks_ts:
        ax2.set_xticks(xticks_ts)
        ax2.set_xticklabels(inputlabels + [f'{n.name}'], rotation=90, fontsize=14)
    else:
        ax2.set_xticks([])

    ax2.xaxis.tick_top()
    ax2.tick_params(which='major', pad=7)
    for tic in ax2.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax2.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax2.set_xlim(-border, ax2width + border)
    ax2.set_ylim(-border, ax2height + border)

    return fig


def render_schemata_png(node):
    """Render one node's schemata once and return a base64 PNG payload."""
    figure = plot_schemata(node)
    buffer = io.BytesIO()
    try:
        figure.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            pad_inches=0.35,
            dpi=200,
            facecolor="white",
        )
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    finally:
        plt.close(figure)


def render_canalization_map_svg(node):
    """Render one node's canalization map and return its payload and size class."""
    try:
        canalization_map = node.canalizing_map(bound="upper")
    except TypeError:
        canalization_map = node.canalizing_map()

    graph = draw_canalizing_map_graphviz(canalization_map)
    graph.graph_attr.update({"pad": "0.02", "margin": "0.0", "ratio": "compress"})
    svg = clean_svg_for_panel(graphviz_svg_from_source(graph.source, graph.engine))
    try:
        schemata_count = len(node.schemata_look_up_table())
    except Exception:
        schemata_count = 3
    return base64.b64encode(svg.encode("utf-8")).decode("utf-8"), schemata_count


# -------------------- UI --------------------
registry, failed_extra = build_model_registry()
all_model_names = sorted(list(registry.keys()), key=lambda x: x.lower())

default_name = "Apoptosis Network" if "Apoptosis Network" in all_model_names else all_model_names[0]

st.sidebar.image(
    CASCI_LOGO_PATH,
    width=120,
)

with st.sidebar.expander("Quick guide", expanded=False):
    st.markdown(
        """
        <style>
        .quick-guide-credit {
            margin-top: 0.85rem;
            color: #64748b;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 0.80rem;
            font-style: italic;
        }
        .quick-guide-credit a { color: #1667b7; }
        .st-key-node-selector-panel { min-height: 76px; }
        </style>
        Explore Boolean-network models through their regulatory structure, Boolean logic, and canalization properties.
        <p>Select a Cell Collective model or upload a <code>.cnet</code> file, adjust the threshold, and click a node to examine its regulators,
        targets, Boolean schemata, canalization map, and node-level parameters.</p>
        <div class="quick-guide-credit">
          Developed at <a href="https://casci.binghamton.edu/casci.php" target="_blank" rel="noopener noreferrer">CASCI Lab</a>
          &nbsp;·&nbsp; Built with <a href="https://github.com/CASCI-lab/CANA" target="_blank" rel="noopener noreferrer">CANA</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

uploaded_cnet = st.sidebar.file_uploader(
    "Upload a .cnet  Boolean network file",
    type=["cnet", "txt"],
    key="uploaded_cnet_file",
    label_visibility="collapsed",
)

use_uploaded = uploaded_cnet is not None

uploaded_bn = None
uploaded_name = None
uploaded_error = None
uploaded_bytes = None

if use_uploaded:
    try:
        uploaded_bytes = uploaded_cnet.getvalue()
        if len(uploaded_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"The file is {len(uploaded_bytes) / (1024 * 1024):.1f} MB; "
                f"the interactive limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            )
        uploaded_bn = load_uploaded_cnet_from_bytes(uploaded_bytes, uploaded_cnet.name)
        validate_uploaded_network(uploaded_bn)
        uploaded_name = get_bn_display_name(uploaded_bn, fallback=uploaded_cnet.name)
        st.sidebar.success(f"Loaded uploaded network: {uploaded_name}")
    except Exception as e:
        uploaded_error = str(e)
        st.sidebar.error("Could not load the uploaded CNET file.")
        st.sidebar.caption(uploaded_error)

if uploaded_error:
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Select model",
    all_model_names,
    index=all_model_names.index(default_name),
    key="model_select",
    disabled=use_uploaded
)

metric = st.sidebar.selectbox(
    "Metric",
    ["Edge effectiveness", "Activity", "Excess canalization", "Correlation"],
    index=0,
    key="metric_select"
)

degree_mode = st.sidebar.toggle("Use in-degree for node coloring", value=False, key="degree_toggle")
degree_mode = "In-degree" if degree_mode else "Out-degree"

if use_uploaded and uploaded_bn is not None:
    bn = uploaded_bn
    source_label = f"Uploaded file: {uploaded_cnet.name}"
else:
    selected_entry = registry[selected_model_name]
    bn = selected_entry["bn"]
    source_label = selected_entry["source"]

if bn is None:
    st.error("No Boolean network could be loaded.")
    st.stop()

bn_name = get_bn_display_name(
    bn,
    fallback=uploaded_cnet.name if (use_uploaded and uploaded_cnet is not None) else selected_model_name
)

current_model_cache_key = model_cache_key(source_label, bn_name, uploaded_bytes)
try:
    SG = session_cached(
        "_network_analysis_cache",
        (current_model_cache_key, "structural_graph"),
        bn.structural_graph,
    )
except Exception as e:
    st.error(f"Could not build the structural graph for this network: {e}")
    st.stop()

N = SG.number_of_nodes()
adaptive_default = float(np.clip(3.0 / max(np.sqrt(max(N, 1)), 1.0), 0.1, 1.0))

EG0 = None
SG_corr = None
edge_activity, edge_excess, edge_corr = {}, {}, {}
act_min, act_max = 0.0, 1.0
ex_min, ex_max = 0.0, 1.0
corr_abs_min, corr_abs_max = 0.0, 1.0

try:
    if metric == "Edge effectiveness":
        EG0 = session_cached(
            "_network_analysis_cache",
            (current_model_cache_key, "effective_graph"),
            bn.effective_graph,
        )
    elif metric in {"Activity", "Excess canalization"}:
        include_excess = metric == "Excess canalization"
        structural_result = session_cached(
            "_network_analysis_cache",
            (current_model_cache_key, "structural_metrics", include_excess),
            lambda: compute_structural_metrics(bn, include_excess=include_excess),
        )
        SG, edge_activity, (act_min, act_max), edge_excess, (ex_min, ex_max) = structural_result
    else:
        SG_corr, edge_corr, (corr_abs_min, corr_abs_max) = session_cached(
            "_network_analysis_cache",
            (current_model_cache_key, "correlation"),
            lambda: compute_correlation_metrics(bn),
        )
except Exception as e:
    st.error(f"Could not compute {metric.lower()} for this network: {e}")
    st.stop()

if metric == "Edge effectiveness":
    thr_min, thr_max = 0.0, 1.0
elif metric == "Activity":
    thr_min, thr_max = float(act_min), float(act_max)
elif metric == "Excess canalization":
    thr_min, thr_max = float(ex_min), float(ex_max)
else:
    thr_min, thr_max = 0.0, float(corr_abs_max)

if thr_min == thr_max:
    thr_min = 0.0

thr_default = thr_min
step = (thr_max - thr_min) / 100.0 if thr_max > thr_min else 0.01


def clear_graph_focus_for_threshold():
    """Return the graph to its overview when edge filtering changes."""
    st.session_state["_graph_focus_node_id"] = ""
    st.session_state["_graph_focus_context"] = ""
    # The component keeps a browser-side focus value too. Changing this token
    # gives it a new context and clears that value on the next render.
    st.session_state["_graph_focus_reset_token"] = (
        int(st.session_state.get("_graph_focus_reset_token", 0)) + 1
    )


thr = st.sidebar.slider(
    "Threshold", float(thr_min), float(thr_max), float(thr_default), float(step),
    key="thr_slider",
    on_change=clear_graph_focus_for_threshold,
)

node_size_in = adaptive_default

node_names = [getattr(node, "name", f"node_{i}") for i, node in enumerate(bn.nodes)]
selected_node_name = st.session_state.get("node_schemata_select", node_names[0])
if selected_node_name not in node_names:
    selected_node_name = node_names[0]
    st.session_state["node_schemata_select"] = selected_node_name
node_selector_panel = st.sidebar.container(key="node-selector-panel")
node_selector_slot = node_selector_panel.empty()

weights = None

if metric == "Edge effectiveness":
    EGf = threshold_graph(EG0, thr)
    special = detect_special_nodes(bn, EG0)
    nodes_order, pos = circular_positions(EG0)
    pos = {n: pos[n] for n in nodes_order}

    isolated = {
        n for n in EGf.nodes()
        if EGf.in_degree(n) == 0 and EGf.out_degree(n) == 0
    }

    node_vals = node_values_from_thresholded_effective(EGf, degree_mode=degree_mode)

    g, max_val = build_graphviz_effective(
        EG0,
        node_vals,
        special,
        pos,
        node_size_in,
        isolated_nodes=isolated
    )
    add_edges_effective(g, EG0, thr, pos)
    graph_source = EG0
    cbar_label = f"Effective {degree_mode.lower()}"

    weights = [float(d.get('weight', 0.0)) for _, _, d in EG0.edges(data=True)]

elif metric == "Activity":
    special = detect_special_nodes(bn, SG)
    nodes_order, pos = circular_positions(SG)
    pos = {n: pos[n] for n in nodes_order}

    node_vals = node_values_from_thresholded_structural(
        SG, edge_activity, thr, degree_mode=degree_mode
    )
    isolated = isolated_nodes_after_threshold(SG, edge_values=edge_activity, thr=thr)

    g, max_val = build_graphviz_structural(
        SG, node_vals, special, pos, node_width_in=node_size_in, isolated_nodes=isolated
    )
    add_edges_structural(
        g, SG, edge_activity, act_min, act_max, pos, thr=thr,
        show_zero_dashed_at_zero_threshold=True,
        signed_color=False,
        threshold_on_abs=False,
        negative_dashed=False
    )
    graph_source = SG
    cbar_label = f"Sum of {degree_mode.lower()} activity"

    weights = list(edge_activity.values())

elif metric == "Excess canalization":
    special = detect_special_nodes(bn, SG)
    nodes_order, pos = circular_positions(SG)
    pos = {n: pos[n] for n in nodes_order}

    node_vals = node_values_from_thresholded_structural(
        SG, edge_excess, thr, degree_mode=degree_mode
    )
    isolated = isolated_nodes_after_threshold(SG, edge_values=edge_excess, thr=thr)

    g, max_val = build_graphviz_structural(
        SG, node_vals, special, pos, node_width_in=node_size_in, isolated_nodes=isolated
    )
    add_edges_structural(
        g, SG, edge_excess, ex_min, ex_max, pos, thr=thr,
        show_zero_dashed_at_zero_threshold=False,
        signed_color=False,
        threshold_on_abs=False,
        negative_dashed=False
    )
    graph_source = SG
    cbar_label = f"Sum of {degree_mode.lower()} excess canalization"

    weights = list(edge_excess.values())

else:
    special = detect_special_nodes(bn, SG_corr)
    nodes_order, pos = circular_positions(SG_corr)
    pos = {n: pos[n] for n in nodes_order}

    node_vals = node_values_from_thresholded_structural(
        SG_corr, edge_corr, thr, degree_mode=degree_mode, use_abs=True
    )
    isolated = isolated_nodes_after_threshold(SG_corr, edge_values=edge_corr, thr=thr, use_abs=True)

    g, max_val = build_graphviz_structural(
        SG_corr, node_vals, special, pos, node_width_in=node_size_in, isolated_nodes=isolated
    )
    add_edges_structural(
        g, SG_corr, edge_corr, corr_abs_min, corr_abs_max, pos, thr=thr,
        show_zero_dashed_at_zero_threshold=True,
        signed_color=True,
        threshold_on_abs=True,
        negative_dashed=True
    )
    graph_source = SG_corr
    cbar_label = f"Sum of |{degree_mode.lower()} correlation|"

    weights = list(edge_corr.values())


model_node_count = SG.number_of_nodes()
model_edge_count = SG.number_of_edges()
model_input_node_count = len(detect_special_nodes(bn, SG))
model_output_node_count = sum(1 for node_id in SG.nodes() if SG.out_degree(node_id) == 0)
graph_node_name_by_id = {
    str(node_id): str(graph_source.nodes[node_id].get("label", node_id))
    for node_id in graph_source.nodes()
}
focus_reset_token = int(st.session_state.get("_graph_focus_reset_token", 0))
graph_focus_context = f"overview-v4|{current_model_cache_key}|{focus_reset_token}"
focused_graph_node_id = (
    st.session_state.get("_graph_focus_node_id", "")
    if st.session_state.get("_graph_focus_context") == graph_focus_context
    else ""
)


st.markdown(f"### {bn_name}")
st.caption(f"Source: {source_label}")

if source_label == "Cell Collective":
    source_info = load_cell_collective_source_info(selected_model_name)
    if source_info.get("error"):
        st.caption("Primary-paper metadata is temporarily unavailable from Cell Collective.")
    else:
        primary_reference = source_info.get("primary_reference")
        paper_link = ""
        if primary_reference and primary_reference.get("url"):
            paper_link = (
                f'<a class="source-paper-link" href="{escape(primary_reference["url"], quote=True)}" '
                'target="_blank" rel="noopener noreferrer">Open primary paper ↗</a>'
            )
        citation = primary_reference["citation"] if primary_reference else "No model-level primary paper is listed."
        citation_html = format_primary_citation(
            citation,
            primary_reference.get("title", "") if primary_reference else "",
        )
        st.markdown(
            f"""
            <style>
            .source-reference-line {{ color: #7a7f89; font-size: 0.875rem; line-height: 1.55; margin: 0.55rem 0 0.95rem; }}
            .source-reference-line a {{ font: inherit; font-weight: 600; text-decoration: none; }}
            .source-reference-line a:hover {{ text-decoration: underline; }}
            .source-primary-label {{ text-decoration: underline; }}
            .source-model-link {{ color: #1667b7; }}
            .source-paper-link {{ color: #a14f22; }}
            .source-reference-separator {{ color: #b4bbc5; padding: 0 0.35rem; }}
            .primary-paper-title {{ font-weight: 750; color: #475569; }}
            </style>
            <div class="source-reference-line">
              <span class="source-primary-label">Primary paper:</span> {citation_html}
              <span class="source-reference-separator">·</span><a class="source-model-link" href="{escape(source_info['model_url'], quote=True)}" target="_blank" rel="noopener noreferrer">View model in Cell Collective ↗</a>
              {f'<span class="source-reference-separator">·</span>{paper_link}' if paper_link else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
elif use_uploaded:
    st.caption("Uploaded CNET model — no external source paper is attached.")

st.markdown(
    """
    <style>
    .dashboard-side-card {
        background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 18px 18px 16px 18px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }
    .dashboard-side-card .section-label,
    .st-key-network-guide .section-label {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .dashboard-side-card .section-title,
    .st-key-network-guide .section-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.8rem;
    }
    .dashboard-side-card .divider,
    .st-key-network-guide .divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(148, 163, 184, 0.12), rgba(148, 163, 184, 0.45), rgba(148, 163, 184, 0.12));
        margin: 16px 0 14px 0;
    }
    .histogram-shell {
        background: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 12px 10px 6px 10px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.85);
    }
    .network-parameters {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 0 0 14px 0;
    }
    .network-parameter {
        padding: 11px 12px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 12px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    }
    .network-parameter-label {
        display: block;
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .network-parameter-value {
        display: block;
        margin-top: 3px;
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.1;
    }
    @media (max-width: 560px) {
        .network-parameters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 1100px) {
        .st-key-network-display [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        .st-key-network-display [data-testid="stColumn"] {
            width: 100% !important;
            min-width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

network_display = st.container(key="network-display")
with network_display:
    c1, c2 = st.columns([3.0, 1.05], gap="medium")

with c1:
    st.markdown(
        f"""
        <div class="network-parameters" aria-label="Model parameters">
          <div class="network-parameter"><span class="network-parameter-label">Nodes</span><span class="network-parameter-value">{model_node_count}</span></div>
          <div class="network-parameter"><span class="network-parameter-label">Edges</span><span class="network-parameter-value">{model_edge_count}</span></div>
          <div class="network-parameter"><span class="network-parameter-label">Input nodes</span><span class="network-parameter-value">{model_input_node_count}</span></div>
          <div class="network-parameter"><span class="network-parameter-label">Output nodes</span><span class="network-parameter-value">{model_output_node_count}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    graph_click_event = render_clickable_network_graph(
        g,
        focused_graph_node_id,
        graph_focus_context,
    )
    if isinstance(graph_click_event, dict):
        click_event_id = str(graph_click_event.get("event_id", ""))
        clicked_node_id = str(graph_click_event.get("node_id", ""))
        clicked_node_name = graph_node_name_by_id.get(clicked_node_id)
        if click_event_id and click_event_id != st.session_state.get("_last_graph_click_event"):
            st.session_state["_last_graph_click_event"] = click_event_id
            st.session_state["_graph_focus_context"] = graph_focus_context
            st.session_state["_graph_focus_node_id"] = clicked_node_id
            if clicked_node_name in node_names and clicked_node_name != selected_node_name:
                st.session_state["node_schemata_select"] = clicked_node_name
                selected_node_name = clicked_node_name

selected_node_name = node_selector_slot.selectbox(
    "Select node for F' / F'' & canalization map",
    node_names,
    key="node_schemata_select",
)
selected_node_index = node_names.index(selected_node_name)
selected_node = bn.nodes[selected_node_index]

with c2:
    with st.container(border=True, key="network-guide"):
        st.markdown('<div class="section-label">Network guide</div>', unsafe_allow_html=True)
        render_graph_legend(metric, thr, degree_mode, max_val)

        if weights:
            st.markdown(
                """
                <div class="divider"></div>
                <div class="section-title" style="margin-bottom:10px; font-weight:800;">Edge value histogram</div>
                """,
                unsafe_allow_html=True
            )

            fig2, ax2 = plt.subplots(figsize=(4.2, 3.35))

            if metric == "Correlation":
                pos_abs = [abs(w) for w in weights if w >= 0]
                neg_abs = [abs(w) for w in weights if w < 0]
                bins = np.linspace(0.0, 1.0, 29)

                ax2.hist(
                    [pos_abs, neg_abs],
                    bins=bins,
                    stacked=True,
                    label=["positive", "negative"]
                )
                ax2.axvline(abs(thr), linestyle="--", color='red', linewidth=2)
                ax2.set_title("Edge correlation", fontsize=10, pad=10)
                ax2.set_xlabel("|Correlation|")
                ax2.set_xlim(0.0, 1.0)
                ax2.legend(frameon=False, fontsize=8)
            else:
                ax2.hist(weights, bins=28)
                ax2.axvline(thr, linestyle="--", color='red', linewidth=2)

                if metric == "Edge effectiveness":
                    ax2.set_title("Edge effectiveness", fontsize=10, pad=10)
                    ax2.set_xlabel("Effectiveness")
                elif metric == "Activity":
                    ax2.set_title("Edge activity", fontsize=10, pad=10)
                    ax2.set_xlabel("Activity")
                elif metric == "Excess canalization":
                    ax2.set_title("Edge excess canalization", fontsize=10, pad=10)
                    ax2.set_xlabel("Excess canalization")

            ax2.set_ylabel("Count")
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(alpha=0.18)

            fig2.tight_layout(pad=1.1)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

selected_node_cache_id = getattr(selected_node, "id", selected_node_index)
(
    selected_node_input_count,
    selected_node_sensitivity,
    selected_node_effective_connectivity,
    selected_node_bias,
) = session_cached(
    "_node_analysis_cache",
    (current_model_cache_key, selected_node_cache_id, "parameters"),
    lambda: (
        int(getattr(selected_node, "k", len(getattr(selected_node, "inputs", []) or []))),
        format_node_parameter(lambda: selected_node.sensitivity(norm=True)),
        format_node_parameter(lambda: selected_node.effective_connectivity(norm=True)),
        format_node_parameter(selected_node.bias),
    ),
)

st.markdown(
    f"""
    <div class="node-parameters-panel" aria-label="Selected node parameters">
      <div class="node-parameters-label">Selected node parameters</div>
      <div class="node-parameters">
        <div class="node-parameter"><span class="node-parameter-label">Inputs</span><span class="node-parameter-value">{selected_node_input_count}</span></div>
        <div class="node-parameter"><span class="node-parameter-label">Sensitivity</span><span class="node-parameter-value">{selected_node_sensitivity}</span></div>
        <div class="node-parameter"><span class="node-parameter-label">Effective connectivity</span><span class="node-parameter-value">{selected_node_effective_connectivity}</span></div>
        <div class="node-parameter"><span class="node-parameter-label">Bias</span><span class="node-parameter-value">{selected_node_bias}</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"#### Node schematas and canalization map for `{selected_node_name}`")

SQUARE_PANEL_SIZE_PX = 700
st.markdown(
    f"""
    <style>
    .square-figure-panel {{
        width: 100%;
        aspect-ratio: 1 / 1;
        max-height: {SQUARE_PANEL_SIZE_PX}px;
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: visible;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 18px;
        box-sizing: border-box;
    }}
    .node-parameters-panel {{
        margin: 1.35rem 0 1rem;
        padding: 14px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 14px;
        background: #fbfcfe;
    }}
    .node-parameters-label {{
        margin-bottom: 10px;
        color: #475569;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}
    .node-parameters {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
    }}
    .node-parameter {{
        min-width: 0;
        padding: 10px 12px;
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 10px;
        background: #ffffff;
    }}
    .node-parameter-label {{
        display: block;
        color: #64748b;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.035em;
        line-height: 1.2;
        min-height: 2.4em;
        text-transform: uppercase;
        white-space: normal;
    }}
    .node-parameter-value {{
        display: block;
        margin-top: 3px;
        color: #0f172a;
        font-size: 1.3rem;
        font-weight: 750;
        line-height: 1.1;
    }}
    @media (max-width: 650px) {{
        .node-parameters {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 850px) {{
        .st-key-node-detail-figures [data-testid="stHorizontalBlock"] {{
            flex-direction: column;
        }}
        .st-key-node-detail-figures [data-testid="stColumn"] {{
            width: 100% !important;
            min-width: 100% !important;
        }}
    }}
    .figure-media-wrap {{
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: visible;
    }}
    .figure-media-wrap img {{
        width: auto;
        height: auto;
        max-width: 100%;
        max-height: 100%;
        display: block;
        margin: auto;
        object-fit: contain;
    }}
    .cmap-panel {{
        padding: 14px;
        background: #fcfcfc;
    }}
    .cmap-panel.half-size-map .figure-media-wrap img {{
        max-width: 50%;
        max-height: 50%;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

node_detail_figures = st.container(key="node-detail-figures")
with node_detail_figures:
    schemata_col, cmap_col = st.columns(2, gap="medium")

with schemata_col:
    st.markdown("##### Node schematas")
    try:
        schemata_b64 = session_cached(
            "_node_analysis_cache",
            (current_model_cache_key, selected_node_cache_id, "schemata_png"),
            lambda: render_schemata_png(selected_node),
        )
        st.markdown(
            f'''
            <div class="square-figure-panel">
                <div class="figure-media-wrap">
                    <img src="data:image/png;base64,{schemata_b64}" alt="Node schematas" />
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not draw F' / F'' schematas for this node: {e}")

with cmap_col:
    st.markdown("##### Canalization map")
    try:
        svg_b64, schemata_count = session_cached(
            "_node_analysis_cache",
            (current_model_cache_key, selected_node_cache_id, "canalization_svg"),
            lambda: render_canalization_map_svg(selected_node),
        )
        half_size_class = " half-size-map" if schemata_count == 2 else ""
        st.markdown(
            f'''
            <div class="square-figure-panel cmap-panel{half_size_class}">
                <div class="figure-media-wrap">
                    <img src="data:image/svg+xml;base64,{svg_b64}" alt="Canalization map" />
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not draw the canalization map for this node: {e}")

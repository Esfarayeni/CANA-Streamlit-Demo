# streamlit2_with_legends.py
# Effective / Activity / Excess Canalization / Correlation Graph Explorer + Schemata Viewer

import os
import io
import base64
import math
import re
import tempfile
from copy import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle, RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.text import Text

import streamlit as st
import graphviz

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


# -------------------- Helpers --------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()


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


@st.cache_resource(show_spinner=True)
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

        g.node(str(n), label(n), pos=f"{x:.3f},{y:.3f}!", color=outline, fillcolor=fill)

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


def compute_structural_metrics(bn):
    SG = bn.structural_graph()
    if SG.number_of_nodes() == 0:
        return SG, {}, (0.0, 1.0), {}, (0.0, 1.0)

    sg_label = {n: SG.nodes[n].get('label', str(n)) for n in SG.nodes()}
    sg_label_norm = {n: _norm(lbl) for n, lbl in sg_label.items()}
    sg_by_label_norm = {lbln: n for n, lbln in sg_label_norm.items()}

    edge_activity = {}
    edge_activity_vals = []
    edge_excess = {}
    edge_excess_vals = []

    for node in bn.nodes:
        name = getattr(node, "name", f"node_{getattr(node, 'id', 'X')}")
        tgt_norm = _norm(name)
        v = sg_by_label_norm.get(tgt_norm, None)
        if v is None:
            continue

        try:
            eff_list = node.edge_effectiveness()
            act_list = node.activities()
        except Exception:
            continue

        if act_list is None:
            continue

        act_list = list(act_list)
        eff_list = list(eff_list) if eff_list is not None else None

        if len(act_list) == 0:
            continue

        preds = list(SG.predecessors(v))
        if len(preds) == 0:
            continue

        L = min(len(preds), len(act_list))
        if L == 0:
            continue

        for i in range(L):
            act = act_list[i]
            if not np.isfinite(act):
                continue

            u = preds[i]
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
def generate_input_columns(k):
    if k <= 0:
        return np.zeros((1, 0), dtype=int)

    rows = []
    for s in range(2 ** k):
        bits = [int(b) for b in format(s, f"0{k}b")]
        rows.append(bits)
    return np.array(rows, dtype=int)


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

    sg_label = {n: SG.nodes[n].get('label', str(n)) for n in SG.nodes()}
    sg_label_norm = {n: _norm(lbl) for n, lbl in sg_label.items()}
    sg_by_label_norm = {lbln: n for n, lbln in sg_label_norm.items()}

    edge_corr = {}
    corr_abs_vals = []

    for node in bn.nodes:
        name = getattr(node, "name", f"node_{getattr(node, 'id', 'X')}")
        tgt_norm = _norm(name)
        v = sg_by_label_norm.get(tgt_norm, None)
        if v is None:
            continue

        inputs = list(getattr(node, "inputs", []) or [])
        k = getattr(node, "k", len(inputs))

        if k <= 0 or len(inputs) == 0:
            continue

        outputs = getattr(node, "outputs", None)
        if outputs is None:
            continue

        outputs = np.asarray(list(outputs), dtype=float)
        if outputs.size != 2 ** k:
            continue

        input_table = generate_input_columns(k)
        preds = list(SG.predecessors(v))

        L = min(len(preds), k, input_table.shape[1])
        for i in range(L):
            u = preds[i]
            col = input_table[:, i]
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

        g.node(str(n), label(n), pos=f"{x:.3f},{y:.3f}!", color=outline, fillcolor=fill)
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
            r'<svg',
            '<svg preserveAspectRatio="xMidYMid meet"',
            svg_text,
            count=1
        )

    if 'class="canalization-map-svg"' not in svg_text:
        svg_text = re.sub(
            r'<svg',
            '<svg class="canalization-map-svg"',
            svg_text,
            count=1
        )

    return svg_text


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


# -------------------- UI --------------------
registry, failed_extra = build_model_registry()
all_model_names = sorted(list(registry.keys()), key=lambda x: x.lower())

default_name = "Apoptosis Network" if "Apoptosis Network" in all_model_names else all_model_names[0]

st.sidebar.image(
    "https://casci.binghamton.edu/img/casci_logo_200.jpg",
    width=120,
)

uploaded_cnet = st.sidebar.file_uploader(
    "Upload a .cnet  Boolean network file",
    type=["cnet", "txt"],
    key="uploaded_cnet_file"
)

use_uploaded = uploaded_cnet is not None

uploaded_bn = None
uploaded_name = None
uploaded_error = None

if use_uploaded:
    try:
        uploaded_bytes = uploaded_cnet.getvalue()
        uploaded_bn = load_uploaded_cnet_from_bytes(uploaded_bytes, uploaded_cnet.name)
        uploaded_name = get_bn_display_name(uploaded_bn, fallback=uploaded_cnet.name)
        st.sidebar.success(f"Loaded uploaded network: {uploaded_name}")
    except Exception as e:
        uploaded_error = str(e)
        st.sidebar.error("Could not load the uploaded CNET file.")
        st.sidebar.caption(uploaded_error)

if failed_extra:
    with st.sidebar.expander("Extra CANA models that failed to load"):
        for k, v in failed_extra.items():
            st.write(f"**{k}**")
            st.caption(v)

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

try:
    EG0 = bn.effective_graph()
except Exception as e:
    st.error(f"Could not build the effective graph for this network: {e}")
    st.stop()

N = EG0.number_of_nodes()
adaptive_default = float(np.clip(3.0 / max(np.sqrt(max(N, 1)), 1.0), 0.1, 1.0))

try:
    SG, edge_activity, (act_min, act_max), edge_excess, (ex_min, ex_max) = compute_structural_metrics(bn)
except Exception as e:
    st.error(f"Could not compute structural metrics for this network: {e}")
    st.stop()

try:
    SG_corr, edge_corr, (corr_abs_min, corr_abs_max) = compute_correlation_metrics(bn)
except Exception as e:
    st.error(f"Could not compute correlation metric for this network: {e}")
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

thr = st.sidebar.slider(
    "Threshold", float(thr_min), float(thr_max), float(thr_default), float(step),
    key="thr_slider"
)

node_size_in = adaptive_default

node_names = [getattr(node, "name", f"node_{i}") for i, node in enumerate(bn.nodes)]
selected_node_name = st.sidebar.selectbox(
    "Select node for F' / F'' & canalization map",
    node_names,
    index=0,
    key="node_schemata_select"
)
selected_node_index = node_names.index(selected_node_name)
selected_node = bn.nodes[selected_node_index]

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
    cbar_label = f"Sum of |{degree_mode.lower()} correlation|"

    weights = list(edge_corr.values())


st.markdown(f"### {bn_name}")
st.caption(f"Source: {source_label}")

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
    .dashboard-side-card .section-label {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .dashboard-side-card .section-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.8rem;
    }
    .dashboard-side-card .divider {
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
    </style>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns([3.0, 1.05], gap="medium")

with c1:
    st.graphviz_chart(g, use_container_width=True)

with c2:
    st.markdown(
        """
        <div class="dashboard-side-card">
            <div class="section-label">Network guide</div>
            <div class="section-title">Legend and distribution</div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"### Node schematas and canalization map for `{selected_node_name}`")

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
    .cmap-inner {{
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: visible;
    }}
    .cmap-inner svg,
    .canalization-map-svg {{
        display: block;
        margin: auto;
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

schemata_col, cmap_col = st.columns(2, gap="medium")

with schemata_col:
    st.markdown("#### Node schematas")
    try:
        schemata_fig = plot_schemata(selected_node)
        buf = io.BytesIO()
        schemata_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.35, dpi=200, facecolor="white")
        buf.seek(0)
        schemata_b64 = base64.b64encode(buf.read()).decode("utf-8")
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
        plt.close(schemata_fig)
    except Exception as e:
        st.warning(f"Could not draw F' / F'' schematas for this node: {e}")

with cmap_col:
    st.markdown("#### Canalization map")
    try:
        try:
            CM = selected_node.canalizing_map(bound='upper')
        except TypeError:
            CM = selected_node.canalizing_map()

        CM_gv = draw_canalizing_map_graphviz(CM)
        try:
            CM_gv.graph_attr.update({
                "pad": "0.02",
                "margin": "0.0",
                "ratio": "compress"
            })
        except Exception:
            pass

        svg = CM_gv.pipe(format="svg").decode("utf-8")
        svg = clean_svg_for_panel(svg)
        svg_b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        st.markdown(
            f'''
            <div class="square-figure-panel cmap-panel">
                <div class="figure-media-wrap">
                    <img src="data:image/svg+xml;base64,{svg_b64}" alt="Canalization map" />
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not draw the canalization map for this node: {e}")

# streamlit2.py
# Effective / Activity / Excess Canalization Graph Explorer

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import graphviz

from cana.datasets.bio import load_all_cell_collective_models

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

DEFAULT_OUTLINE = "#ff9896"
SPECIAL_OUTLINE = "#ffdf0e"

cmap = LinearSegmentedColormap.from_list('custom', ['white', '#d62728'])
cmap.set_under('#2ca02c')  # nodes with zero value → green


def _norm(s: str) -> str:
    return str(s).strip().lower()


@st.cache_resource(show_spinner=True)
def load_models():
    return list(load_all_cell_collective_models())


def list_model_names(models):
    return [m.name.strip() for m in models]


def get_model_by_name(models, target_name: str):
    tgt = target_name.strip().lower()
    for m in models:
        if m.name.strip().lower() == tgt:
            return m
    for m in models:
        if tgt in m.name.strip().lower():
            return m
    return None


def detect_special_nodes(bn, G):
    """Detect k=0 or k=1 self-loop nodes, mapped to graph G by label."""
    special_norm_labels = set()
    for node in bn.nodes:
        name   = getattr(node, "name", f"node_{getattr(node, 'id', 'X')}")
        inputs = list(getattr(node, "inputs", []) or [])
        k      = getattr(node, "k", len(inputs))

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
    label = lambda n: G.nodes[n].get('label', str(n))
    sorted_nodes = sorted(nodes, key=label)
    pos = {n: (radius*np.cos(2*np.pi*i/N), radius*np.sin(2*np.pi*i/N))
           for i, n in enumerate(sorted_nodes)}
    return sorted_nodes, pos


def threshold_graph(EG0, thr):
    """Remove edges with w ≤ thr (keep only w > thr) on effective graph."""
    EGf = EG0.copy()
    EGf.remove_edges_from([
        (u, v) for u, v, d in EGf.edges(data=True)
        if float(d.get('weight', 0.0)) <= thr
    ])
    return EGf


def colorbar_figure(max_val, label_text):
    """Generic vertical colorbar for node values."""
    fig = plt.figure(figsize=(0.38, 0.80))
    ax = fig.add_axes([0.45, 0.12, 0.12, 0.70])

    ticks_num = min(4, int(round(max_val)) + 1)
    ticks = np.linspace(0, max_val, num=max(2, ticks_num)).round().astype(int)
    norm_mpl = mpl.colors.Normalize(vmin=1e-16, vmax=max_val if max_val > 0 else 1)

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm_mpl,
        ticks=ticks,
        orientation='vertical',
        spacing='uniform',
        extend='min',
        format='%d'
    )
    # Smaller legend text
    cb.set_label(label_text, fontsize=4)
    cb.ax.tick_params(labelsize=4)
    return fig


# ---------- Effective-graph (edge effectiveness) visualization ----------

def build_graphviz_effective(EG0, EGf, special_nodes_set, positions, node_width_in):
    outdeg_f = {n: EGf.out_degree(n, weight='weight') for n in EGf.nodes()}
    for n in EG0.nodes():
        outdeg_f.setdefault(n, 0.0)
    max_outdeg_f = max(outdeg_f.values()) if outdeg_f else 1.0
    norm_mpl = mpl.colors.Normalize(vmin=1e-16, vmax=max_outdeg_f if max_outdeg_f > 0 else 1)

    g = graphviz.Digraph(engine='neato')
    g.attr(
        'graph',
        size=f'{CANVAS_INCH},{CANVAS_INCH}!',
        ratio='1',
        margin='0.2',
        pad='0.1',
        splines='True',      
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
        out_d = outdeg_f.get(n, 0.0)
        if out_d == 0:
            fill, outline = '#2ca02c', '#98df8a'
        else:
            fill = mpl.colors.rgb2hex(cmap(norm_mpl(out_d)))
            outline = SPECIAL_OUTLINE if n in special_nodes_set else DEFAULT_OUTLINE
        g.node(str(n), label(n), pos=f"{x:.3f},{y:.3f}!", color=outline, fillcolor=fill)
    return g, max_outdeg_f


def add_edges_effective(g, EG0, thr):
    for u, v, d in EG0.edges(data=True):
        w = float(d.get('weight', 0.0))
        if w <= thr:
            continue
        pen = max(0.5, min(PENWIDTH_MAX, PENWIDTH_MAX * w)) if w > 0 else 1.5
        g.edge(str(u), str(v), penwidth=f"{pen:.2f}")


# ---------- Structural metrics: Activity & Excess canalization ----------

def metric_to_width(val, vmin, vmax, wmin=MIN_WIDTH, wmax=MAX_WIDTH):
    """Map metric to edge width."""
    if vmax > vmin:
        t = (val - vmin) / (vmax - vmin)
    else:
        t = 0.5
    return wmin + t * (wmax - wmin)


def compute_structural_metrics(bn):
    """Compute edge activity and edge excess (effectiveness - activity) on SG."""
    SG = bn.structural_graph()
    if SG.number_of_nodes() == 0:
        return SG, {}, (0.0, 1.0), {}, 1.0, {}, (0.0, 1.0), {}, 1.0

    # Map SG node IDs <-> normalized labels
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

            # excess = effectiveness - activity, if availability allows
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

    # Aggregate outgoing values per node (full, not thresholded yet)
    node_out_activity = {n: 0.0 for n in SG.nodes()}
    for (u, v), val in edge_activity.items():
        node_out_activity[u] = node_out_activity.get(u, 0.0) + max(0.0, val)
    max_out_activity = max(node_out_activity.values()) if node_out_activity else 1.0

    node_out_excess = {n: 0.0 for n in SG.nodes()}
    for (u, v), val in edge_excess.items():
        node_out_excess[u] = node_out_excess.get(u, 0.0) + max(0.0, val)
    max_out_excess = max(node_out_excess.values()) if node_out_excess else 1.0

    return (
        SG,
        edge_activity, (act_min, act_max), node_out_activity, max_out_activity,
        edge_excess, (ex_min, ex_max), node_out_excess, max_out_excess
    )


def build_graphviz_structural(SG, node_values, special_nodes_set, positions, node_width_in):
    """Build Graphviz for structural metrics (activity / excess)."""
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
        splines='True',

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
            fill, outline = '#2ca02c', '#98df8a'
        else:
            fill = mpl.colors.rgb2hex(cmap(norm_mpl(val)))
            outline = SPECIAL_OUTLINE if n in special_nodes_set else DEFAULT_OUTLINE
        g.node(str(n), label(n), pos=f"{x:.3f},{y:.3f}!", color=outline, fillcolor=fill)
    return g, max_val


def add_edges_structural(g, SG, edge_values, vmin, vmax, thr=None):
    for u, v, d in SG.edges(data=True):
        key = (u, v)
        if key in edge_values:
            val = edge_values[key]
            # remove edges with value <= threshold
            if thr is not None and val <= thr:
                continue
            width = metric_to_width(val, vmin, vmax)
            g.edge(str(u), str(v), penwidth=f"{width:.2f}", color="black")
        else:
            # gray, default-width edge when no metric is available
            g.edge(str(u), str(v), penwidth="1.0", color="#bbbbbb")


# -------------------- UI --------------------
models = load_models()
names = list_model_names(models)
default_name = "MAPK Cancer Cell Fate Network" if "MAPK Cancer Cell Fate Network" in names else names[0]

st.sidebar.header("Controls")
target_name = st.sidebar.selectbox("Select model", names, index=names.index(default_name), key="model_select")

metric = st.sidebar.selectbox(
    "Metric",
    ["Edge effectiveness", "Activity", "Excess canalization"],
    index=0,
    key="metric_select"
)

bn = get_model_by_name(models, target_name)

# Effective graph
EG0 = bn.effective_graph()
N = EG0.number_of_nodes()

adaptive_default = float(np.clip(3.0 / max(np.sqrt(N), 1.0), 0.1, 1.0))

# Structural metrics (needed for threshold ranges of Activity/Excess)
(
    SG,
    edge_activity, (act_min, act_max), node_out_activity, max_out_activity,
    edge_excess, (ex_min, ex_max), node_out_excess, max_out_excess
) = compute_structural_metrics(bn)

# -------- Threshold slider depends on chosen metric --------
if metric == "Edge effectiveness":
    thr_min, thr_max = 0.0, 1.0
elif metric == "Activity":
    thr_min, thr_max = float(act_min), float(act_max)
else:  # Excess canalization
    thr_min, thr_max = float(ex_min), float(ex_max)

# Avoid degenerate slider if min == max
if thr_min == thr_max:
    thr_min = 0.0

thr_default = thr_min
step = (thr_max - thr_min) / 100.0 if thr_max > thr_min else 0.01

thr = st.sidebar.slider(
    "Threshold", float(thr_min), float(thr_max), float(thr_default), float(step),
    key="thr_slider"
)

node_size_key = f"node_size_in_{target_name}"
node_size_in = st.sidebar.slider("Node size (inches)", 0.1, 1.0, adaptive_default, 0.05, key=node_size_key)
st.sidebar.caption(f"Default for N={N}: {adaptive_default:.2f} in")

# -------------------- Build graphs based on metric --------------------
if metric == "Edge effectiveness":
    EGf = threshold_graph(EG0, thr)
    special = detect_special_nodes(bn, EG0)
    nodes_order, pos = circular_positions(EG0)
    pos = {n: pos[n] for n in nodes_order}
    g, max_val = build_graphviz_effective(EG0, EGf, special, pos, node_size_in)
    add_edges_effective(g, EG0, thr)
    cbar_label = "Effective Out-degree"

elif metric in ("Activity", "Excess canalization"):
    # Use structural graph
    special = detect_special_nodes(bn, SG)
    nodes_order, pos = circular_positions(SG)
    pos = {n: pos[n] for n in nodes_order}

    if metric == "Activity":
        # Node values based on edges with value > threshold
        node_vals = {n: 0.0 for n in SG.nodes()}
        for (u, v), val in edge_activity.items():
            if val > thr:
                node_vals[u] += max(0.0, val)
        g, max_val = build_graphviz_structural(SG, node_vals, special, pos, node_width_in=node_size_in)
        add_edges_structural(g, SG, edge_activity, act_min, act_max, thr=thr)
        cbar_label = "Sum of outgoing activity"
    else:  # Excess canalization
        node_vals = {n: 0.0 for n in SG.nodes()}
        for (u, v), val in edge_excess.items():
            if val > thr:
                node_vals[u] += max(0.0, val)
        g, max_val = build_graphviz_structural(SG, node_vals, special, pos, node_width_in=node_size_in)
        add_edges_structural(g, SG, edge_excess, ex_min, ex_max, thr=thr)
        cbar_label = "Sum of outgoing excess canalization"


# -------------------- Display --------------------
st.markdown(f"### {bn.name}")

if metric == "Edge effectiveness":
    desc = (
        "Edges with value ≤ threshold are removed; "
        "nodes are colored by effective out-degree computed from remaining edges."
    )
elif metric == "Activity":
    desc = (
        "Nodes and edges are styled by input activity: edges with activity ≤ threshold are removed, "
        "thicker edges indicate higher remaining activity, and node color reflects the sum of "
        "outgoing activity over edges above the threshold."
    )
else:  # Excess canalization
    desc = (
        "Nodes and edges are styled by excess canalization (edge effectiveness − activity): "
        "edges with excess ≤ threshold are removed, thicker edges indicate larger remaining excess, "
        "and node color reflects the sum of outgoing excess over edges above the threshold."
    )

st.markdown(
    f"<p style='font-size:18px; font-weight:500; margin-top:4px;'>{desc}</p>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([3.5, 1.0], gap="small")

with c1:
    st.graphviz_chart(g, use_container_width=True)
with c2:
    st.pyplot(colorbar_figure(max_val, cbar_label))

# -------------------- Histogram (only for effective graph) --------------------
if metric == "Edge effectiveness":
    with st.expander("Edge-weight histogram (before thresholding)", expanded=False):
        weights = [float(d.get('weight', 0.0)) for _, _, d in EG0.edges(data=True)]
        if weights:
            fig2, ax2 = plt.subplots(figsize=(4, 2.4))
            ax2.hist(weights, bins=24)
            ax2.axvline(thr, linestyle="--", color='red')
            ax2.set_xlabel("Value")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)
        else:
            st.info("No edges found.")

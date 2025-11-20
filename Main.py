# streamlit2.py
# Effective / Activity / Excess Canalization Graph Explorer + Schemata Viewer

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle, RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.text import Text
from copy import copy

import streamlit as st
import graphviz

from cana.datasets.bio import load_all_cell_collective_models
from cana.drawing.canalizing_map import draw_canalizing_map_graphviz
# NOTE: assuming draw_canalizing_map_graphviz is defined/imported elsewhere in this file,
# as in your notebook snippet:
# from some_module import draw_canalizing_map_graphviz

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
ISOLATED_OUTLINE  = "#888888"   # outline color for isolated nodes (no in/out after threshold)

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
    # Colorbar style
    cmap = LinearSegmentedColormap.from_list('custom', ['white', '#d62728'])
    cmap.set_under('#2ca02c')

    norm = mpl.colors.Normalize(vmin=1e-16, vmax=max_val if max_val > 0 else 1)

    # Slightly larger but still small figure, with good resolution
    fig = plt.figure(figsize=(0.8, 1.6), dpi=200)
    #          width   height (inches)

    # Leave enough room for label + ticks
    ax = fig.add_axes([0.03, 0.05, 0.15, 0.6])
    #      left bottom width height

    # Ticks
    tick_max = int(max_val)
    ticks = list(range(0, tick_max + 1, max(1, tick_max // 4 or 1)))
    boundaries = np.linspace(-1, max_val, 25).tolist()

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
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


# ---------- Helper: detect isolated nodes after threshold ----------

def isolated_nodes_after_threshold(G, edge_values=None, thr=None):
    """
    Return nodes with in-degree == 0 and out-degree == 0
    in the thresholded graph (according to the metric/threshold).
    For structural graphs:
      - if edge_values[key] exists and val <= thr -> remove edge
      - if no metric for edge -> keep edge (as in add_edges_structural)
    """
    H = G.copy()
    if edge_values is not None and thr is not None:
        to_remove = []
        for u, v in H.edges():
            key = (u, v)
            if key in edge_values:
                val = edge_values[key]
                if val <= thr:
                    to_remove.append((u, v))
        H.remove_edges_from(to_remove)

    return {
        n for n in H.nodes()
        if H.in_degree(n) == 0 and H.out_degree(n) == 0
    }


# ---------- Effective-graph (edge effectiveness) visualization ----------

def build_graphviz_effective(EG0, EGf, special_nodes_set, positions, node_width_in, isolated_nodes=None):
    if isolated_nodes is None:
        isolated_nodes = set()

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
            fill = '#2ca02c'
        else:
            fill = mpl.colors.rgb2hex(cmap(norm_mpl(out_d)))

        # outline logic: isolated → gray; else special/default
        if n in isolated_nodes:
            outline = ISOLATED_OUTLINE
        elif n in special_nodes_set:
            outline = SPECIAL_OUTLINE
        else:
            outline = DEFAULT_OUTLINE

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


def build_graphviz_structural(SG, node_values, special_nodes_set, positions, node_width_in, isolated_nodes=None):
    """Build Graphviz for structural metrics (activity / excess)."""
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
            fill = '#2ca02c'
        else:
            fill = mpl.colors.rgb2hex(cmap(norm_mpl(val)))

        # outline logic: isolated → gray; else special/default
        if n in isolated_nodes:
            outline = ISOLATED_OUTLINE
        elif n in special_nodes_set:
            outline = SPECIAL_OUTLINE
        else:
            outline = DEFAULT_OUTLINE

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


# ---------- Schemata plotting (F' and F'') ----------

def plot_schemata(n):
    """
    Plot F' and F'' schematas for a CANA BooleanNode `n`
    with a fixed small figure size (for Streamlit).
    Returns a matplotlib Figure.
    """
    # ---------- Ensure schemata are computed ----------
    if (getattr(n, "_prime_implicants", None) is None or
        getattr(n, "_two_symbols", None) is None):
        if hasattr(n, "schemata"):
            try:
                n.schemata()
            except Exception:
                pass

    pi_dict = getattr(n, "_prime_implicants", {}) or {}
    two = getattr(n, "_two_symbols", None)

    pi0s = pi_dict.get('0', []) or []
    pi1s = pi_dict.get('1', []) or []
    if two is None:
        ts0s, ts1s = [], []
    else:
        try:
            ts0s = two[0] or []
            ts1s = two[1] or []
        except Exception:
            ts0s, ts1s = [], []

    # Basic parameters
    k = n.k if n.k >= 1 else 1
    inputs = n.inputs if not n.constant else [n.name]
    inputlabels = [n.network.get_node_name(i)[0] if n.network is not None else i for i in inputs]

    # We still compute these for internal layout, but we DO NOT use them for figsize
    n_pi = sum(len(pis) for pis in [pi0s, pi1s])
    n_ts = sum(len(tss) for tss in [ts0s, ts1s])

    # ---------- FIXED small figure size ----------
    fig_width, fig_height = 3.0, 1.6   # <<– make this smaller/bigger if you want
    # --------------------------------------------

    cwidth = 60.
    cxspace = 0
    cyspace = 6
    border = 1
    sepcxspace = 21
    sepcyspace = 15

    ax1width  = (k * (cwidth + cxspace)) + sepcxspace + cwidth
    ax1height = (max(n_pi, 1) * (cwidth + cyspace)) + sepcyspace - cyspace
    ax2width  = (k * (cwidth + cxspace)) + sepcxspace + cwidth
    ax2height = (max(n_ts, 1) * (cwidth + cyspace)) + sepcyspace - cyspace

    # ---------- Figure + subplots ----------
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1, 1],
        left=0.05, right=0.98, bottom=0.2, top=0.85, wspace=0.25
    )
    ax1 = fig.add_subplot(gs[0, 0])  # F'
    ax2 = fig.add_subplot(gs[0, 1])  # F''

    # ===================== F' (prime implicants) =====================
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    from matplotlib.text import Text

    yticks = []
    patches = []
    x, y = 0., 0.

    for out, pis in zip([1, 0], [pi1s, pi0s]):
        for pi in pis:
            x = 0.
            xticks = []
            for inp in pi:
                if inp == '0':
                    facecolor = 'white'
                    textcolor = 'black'
                elif inp == '1':
                    facecolor = 'black'
                    textcolor = 'white'
                elif inp == '#':
                    facecolor = '#cccccc'
                    textcolor = 'black'
                else:
                    facecolor = 'white'
                    textcolor = 'black'

                text = inp if inp != '2' else '#'
                ax1.add_artist(Text(
                    x + cwidth / 2, y + cwidth * 0.4,
                    text=text, color=textcolor,
                    va='center', ha='center',
                    fontsize=10,  # slightly smaller
                    family='serif'
                ))
                r = Rectangle(
                    (x, y), width=cwidth, height=cwidth,
                    facecolor=facecolor, edgecolor='black'
                )
                patches.append(r)
                xticks.append(x + cwidth / 2)
                x += cwidth + cxspace

            x += sepcxspace
            r = Rectangle(
                (x, y), width=cwidth, height=cwidth,
                facecolor='black' if out == 1 else 'white',
                edgecolor='black'
            )
            ax1.add_artist(Text(
                x - (sepcxspace / 2) - (cxspace / 2),
                y + cwidth * 0.4,
                text=':', color='black',
                va='center', ha='center',
                fontsize=10, weight='bold', family='serif'
            ))
            ax1.add_artist(Text(
                x + cwidth / 2, y + cwidth * 0.4,
                text=str(out),
                color='white' if out == 1 else 'black',
                va='center', ha='center',
                fontsize=10, family='serif'
            ))
            patches.append(r)
            xticks.append(x + cwidth / 2)
            yticks.append(y + cwidth / 2)
            y += cwidth + cyspace
        y += sepcyspace

    if patches:
        ax1.add_collection(PatchCollection(patches, match_original=True))

    if yticks:
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(
            [rf"$f^{{'}}_{{{i+1}}}$" for i in range(len(yticks))[::-1]],
            fontsize=8
        )
    else:
        ax1.set_yticks([])

    if 'xticks' in locals() and xticks:
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(inputlabels + [str(n.name)], rotation=90, fontsize=8)
    else:
        ax1.set_xticks([])

    ax1.set_xlim(-border, ax1width + border)
    ax1.set_ylim(-border, ax1height + border)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    ax1.tick_params(which='major', pad=4)
    for tic in ax1.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax1.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for side in ['top', 'right', 'bottom', 'left']:
        ax1.spines[side].set_visible(False)
    ax1.set_title("F' schematas", fontsize=10, pad=6)

    # ===================== F'' (two-symbol schematas) =====================
    from matplotlib.patches import Circle, RegularPolygon
    yticks = []
    boxes, symbols = [], []
    x, y = 0., 0.

    tssymbols = [
        Circle((0, 0), radius=5, facecolor='white', edgecolor='black'),
        RegularPolygon((0, 0), numVertices=3, radius=5, orientation=0,
                       facecolor='white', edgecolor='black'),
    ]

    for out, tss in zip([1, 0], [ts1s, ts0s]):
        for ts, pss, sss in tss:
            x = 0.
            xticks = []
            for i, inp in enumerate(ts):
                if inp == '0':
                    facecolor = 'white'
                    textcolor = 'black'
                elif inp == '1':
                    facecolor = 'black'
                    textcolor = 'white'
                elif inp == '2':
                    facecolor = '#cccccc'
                    textcolor = 'black'
                else:
                    facecolor = 'white'
                    textcolor = 'black'

                if pss:
                    iinpss = [j for j, ps in enumerate(pss) if i in ps]
                    xpos = np.linspace(x, x + cwidth, len(iinpss) + 2)
                    for z, j in enumerate(iinpss, start=1):
                        if j >= len(tssymbols):
                            continue
                        s = copy(tssymbols[j])
                        s.xy = (xpos[z], y + cwidth * 0.8)
                        s.center = (xpos[z], y + cwidth * 0.8)
                        s.set_edgecolor('#a6a6a6' if inp == '1' else 'black')
                        symbols.append(s)
                        ax2.add_patch(s)

                text = inp if inp != '2' else '#'
                ax2.add_artist(Text(
                    x + cwidth / 2, y + cwidth * 0.4,
                    text=text, color=textcolor,
                    va='center', ha='center',
                    fontsize=10, family='serif'
                ))
                r = Rectangle(
                    (x, y), width=cwidth, height=cwidth,
                    facecolor=facecolor, edgecolor='#4c4c4c', zorder=2
                )
                boxes.append(r)
                xticks.append(x + cwidth / 2)
                x += cwidth + cxspace

            x += sepcxspace
            r = Rectangle(
                (x, y), width=cwidth, height=cwidth,
                facecolor='black' if out == 1 else 'white',
                edgecolor='#4c4c4c'
            )
            ax2.add_artist(Text(
                x - (sepcxspace / 2) - (cxspace / 2),
                y + cwidth / 2,
                text=':',
                color='black', va='center', ha='center',
                fontsize=10, weight='bold', family='serif'
            ))
            ax2.add_artist(Text(
                x + cwidth / 2, y + cwidth * 0.4,
                text=str(out),
                color='white' if out == 1 else 'black',
                va='center', ha='center',
                fontsize=10, family='serif'
            ))
            boxes.append(r)
            xticks.append(x + cwidth / 2)
            yticks.append(y + cwidth / 2)
            y += cwidth + cyspace
        y += sepcyspace

    if boxes:
        ax2.add_collection(PatchCollection(boxes, match_original=True))
    if symbols:
        ax2.add_collection(PatchCollection(symbols, match_original=True))

    if yticks:
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(
            [rf"$f^{{''}}_{{{i+1}}}$" for i in range(len(yticks))[::-1]],
            fontsize=8
        )
    else:
        ax2.set_yticks([])

    if 'xticks' in locals() and xticks:
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(inputlabels + [str(n.name)], rotation=90, fontsize=8)
    else:
        ax2.set_xticks([])

    ax2.set_xlim(-border, ax2width + border)
    ax2.set_ylim(-border, ax2height + border)
    ax2.invert_yaxis()
    ax2.xaxis.tick_top()
    ax2.tick_params(which='major', pad=4)
    for tic in ax2.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax2.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for side in ['top', 'right', 'bottom', 'left']:
        ax2.spines[side].set_visible(False)
    ax2.set_title("F'' schematas", fontsize=10, pad=6)

    return fig



# -------------------- UI --------------------
models = load_models()
names = list_model_names(models)
default_name = "Apoptosis Network" if "Apoptosis Network" in names else names[0]

# Sidebar: logo + controls
st.sidebar.image(
    "https://casci.binghamton.edu/img/casci_logo_200.jpg",
    width=120,  # smaller logo
)

st.sidebar.header("Controls")
target_name = st.sidebar.selectbox("Select model", names, index=names.index(default_name), key="model_select")

metric = st.sidebar.selectbox(
    "Metric",
    ["Edge effectiveness", "Activity", "Excess canalization"],
    index=0,
    key="metric_select"
)

bn = get_model_by_name(models, target_name)

# Effective graph & structural metrics
EG0 = bn.effective_graph()
N = EG0.number_of_nodes()

adaptive_default = float(np.clip(3.0 / max(np.sqrt(N), 1.0), 0.1, 1.0))

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

# ----- Node dropdown AFTER sliders (for F', F'' + canalization map) -----
node_names = [getattr(node, "name", f"node_{i}") for i, node in enumerate(bn.nodes)]
selected_node_name = st.sidebar.selectbox(
    "Select node for F' / F'' & canalization map",
    node_names,
    index=0,
    key="node_schemata_select"
)
selected_node_index = node_names.index(selected_node_name)
selected_node = bn.nodes[selected_node_index]

# -------------------- Build graphs based on metric --------------------
weights = None  # for histogram (all metrics now)

if metric == "Edge effectiveness":
    EGf = threshold_graph(EG0, thr)
    special = detect_special_nodes(bn, EG0)
    nodes_order, pos = circular_positions(EG0)
    pos = {n: pos[n] for n in nodes_order}

    # isolated nodes after threshold (no in/out edges in EGf)
    isolated = {
        n for n in EGf.nodes()
        if EGf.in_degree(n) == 0 and EGf.out_degree(n) == 0
    }

    g, max_val = build_graphviz_effective(EG0, EGf, special, pos, node_size_in, isolated_nodes=isolated)
    add_edges_effective(g, EG0, thr)
    cbar_label = "Effective Out-degree"

    # collect weights for histogram (all edges)
    weights = [float(d.get('weight', 0.0)) for _, _, d in EG0.edges(data=True)]

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

        # isolated nodes after threshold for activity
        isolated = isolated_nodes_after_threshold(SG, edge_values=edge_activity, thr=thr)

        g, max_val = build_graphviz_structural(
            SG, node_vals, special, pos, node_width_in=node_size_in, isolated_nodes=isolated
        )
        add_edges_structural(g, SG, edge_activity, act_min, act_max, thr=thr)
        cbar_label = "Sum of outgoing activity"

        # histogram data: all activity values
        weights = list(edge_activity.values())

    else:  # Excess canalization
        node_vals = {n: 0.0 for n in SG.nodes()}
        for (u, v), val in edge_excess.items():
            if val > thr:
                node_vals[u] += max(0.0, val)

        # isolated nodes after threshold for excess
        isolated = isolated_nodes_after_threshold(SG, edge_values=edge_excess, thr=thr)

        g, max_val = build_graphviz_structural(
            SG, node_vals, special, pos, node_width_in=node_size_in, isolated_nodes=isolated
        )
        add_edges_structural(g, SG, edge_excess, ex_min, ex_max, thr=thr)
        cbar_label = "Sum of outgoing excess canalization"

        # histogram data: all excess values
        weights = list(edge_excess.values())


# -------------------- Description --------------------
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
        "outgoing activity over edges above the threshold. Nodes with no incoming or outgoing "
        "edges after thresholding are outlined in gray."
    )
else:  # Excess canalization
    desc = (
        "Nodes and edges are styled by excess canalization (edge effectiveness − activity): "
        "edges with excess ≤ threshold are removed, thicker edges indicate larger remaining excess, "
        "and node color reflects the sum of outgoing excess over edges above the threshold. "
        "Nodes with no incoming or outgoing edges after thresholding are outlined in gray."
    )

st.markdown(
    f"<p style='font-size:18px; font-weight:500; margin-top:4px;'>{desc}</p>",
    unsafe_allow_html=True,
)

# -------------------- Layout: graph left, legend + histogram right --------------------
c1, c2 = st.columns([3.0, 1.0], gap="small")

with c1:
    st.graphviz_chart(g, use_container_width=True)

with c2:
    # Legend (colorbar)
    st.pyplot(colorbar_figure(max_val, cbar_label))

    # Histogram directly below legend, same column width
    if weights:
        fig2, ax2 = plt.subplots(figsize=(2.5, 2.5))
        ax2.hist(weights, bins=24)
        ax2.axvline(thr, linestyle="--", color='red')

        if metric == "Edge effectiveness":
            ax2.set_title("Edge effectiveness (all edges)", fontsize=8)
            ax2.set_xlabel("Effectiveness")
        elif metric == "Activity":
            ax2.set_title("Edge activity (all edges)", fontsize=8)
            ax2.set_xlabel("Activity")
        else:
            ax2.set_title("Edge excess canalization (all edges)", fontsize=8)
            ax2.set_xlabel("Excess canalization")

        ax2.set_ylabel("Count")
        st.pyplot(fig2)

# -------------------- Node schemata + canalization map section --------------------
st.markdown(f"### Node schematas and canalization map for `{selected_node_name}`")

# F' / F'' schematas
schemata_fig = plot_schemata(selected_node)
st.pyplot(schemata_fig)
plt.close(schemata_fig)

# Canalization map (single-node CM)
st.markdown("#### Canalization map")

try:
    CM = selected_node.canalizing_map(bound='upper')
except TypeError:
    # fallback if this version of CANA does not take 'bound'
    CM = selected_node.canalizing_map()

CM_gv = draw_canalizing_map_graphviz(CM)

# If CM_gv is a graphviz.Source or Digraph, this works:
st.graphviz_chart(CM_gv)

# Effective / Activity / Excess Canalization / Correlation Graph Explorer + Schemata Viewer

import hashlib
from html import escape

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from constants import (
    APP_ICON_PATH, APP_TITLE, CASCI_LOGO_PATH, MAX_UPLOAD_BYTES,
    SESSION_CACHE_MAX_ENTRIES,
)
from ui_components import render_quick_guide


# -------------------- Page & style --------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=str(APP_ICON_PATH),
    layout="wide",
)

# Legacy helper implementations were extracted into focused modules.


def session_cached(namespace, key, factory):
    """Keep mutable CANA results in a bounded cache for this browser session."""
    cache = st.session_state.setdefault(namespace, {})
    if key not in cache:
        if len(cache) >= SESSION_CACHE_MAX_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = factory()
    return cache[key]


def model_cache_key(source_label, model_name, uploaded_bytes=None):
    """Use content addressing for uploads and stable keys for built-in models."""
    if uploaded_bytes is not None:
        return f"upload:{hashlib.sha256(uploaded_bytes).hexdigest()}"
    return f"builtin:{source_label}:{model_name}"


# -------------------- UI --------------------
from model_data import (
    build_model_registry,
    format_primary_citation,
    get_bn_display_name,
    load_cell_collective_source_info,
    load_uploaded_cnet_from_bytes,
    validate_uploaded_network,
)
from metrics import (
    compute_correlation_metrics,
    compute_structural_metrics,
    detect_special_nodes,
    isolated_nodes_after_threshold,
    node_values_from_thresholded_effective,
    node_values_from_thresholded_structural,
    threshold_graph,
)
from graph_rendering import (
    add_edges_effective,
    add_edges_structural,
    build_graphviz_effective,
    build_graphviz_structural,
    circular_positions,
    render_clickable_network_graph,
    render_graph_legend,
)
from node_analysis import (
    format_node_parameter,
    render_canalization_map_svg,
    render_schemata_png,
)

registry, failed_extra = build_model_registry()
all_model_names = sorted(list(registry.keys()), key=lambda x: x.lower())

default_name = "Apoptosis Network" if "Apoptosis Network" in all_model_names else all_model_names[0]

st.sidebar.image(
    CASCI_LOGO_PATH,
    width=270,
)

render_quick_guide()

# The uploader is rendered after the analysis controls below. Its keyed value
# is already available at the beginning of a rerun, letting uploads continue
# to take priority over the selected catalogue model.
uploaded_cnet = st.session_state.get("uploaded_cnet_file")
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
    except Exception as e:
        uploaded_error = str(e)


def clear_graph_focus():
    """Return the graph to its overview after a view-defining control changes."""
    st.session_state["_graph_focus_node_id"] = ""
    st.session_state["_graph_focus_context"] = ""
    # The component keeps a browser-side focus value too. Changing this token
    # gives it a new context and clears that value on the next render.
    st.session_state["_graph_focus_reset_token"] = (
        int(st.session_state.get("_graph_focus_reset_token", 0)) + 1
    )


def focus_node_selected_in_sidebar():
    """Request graph focus for an explicit sidebar node-selector change."""
    st.session_state["_sidebar_node_focus_pending"] = True


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
    key="metric_select",
    on_change=clear_graph_focus,
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
    thr_min, thr_max = 0.0, max(0.0, float(act_max))
elif metric == "Excess canalization":
    thr_min, thr_max = 0.0, max(0.0, float(ex_max))
else:
    thr_min, thr_max = 0.0, float(corr_abs_max)

if thr_min == thr_max:
    thr_min = 0.0

thr_default = thr_min
step = (thr_max - thr_min) / 100.0 if thr_max > thr_min else 0.01


thr = st.sidebar.slider(
    "Threshold", float(thr_min), float(thr_max), float(thr_default), float(step),
    key="thr_slider",
    on_change=clear_graph_focus,
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

# The sidebar callback marks deliberate selector changes before this script
# runs. It avoids treating an initial render, model change, or graph click as
# an instruction to alter the graph view.
if st.session_state.pop("_sidebar_node_focus_pending", False):
    selected_graph_node_id = next(
        (
            node_id
            for node_id, node_name in graph_node_name_by_id.items()
            if node_name == selected_node_name
        ),
        "",
    )
    if selected_graph_node_id:
        st.session_state["_graph_focus_context"] = graph_focus_context
        st.session_state["_graph_focus_node_id"] = selected_graph_node_id

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
        flex: 1 1 auto;
        margin: 0;
    }
    .model-parameters-row,
    .node-parameters-panel {
        display: flex;
        align-items: stretch;
        gap: 14px;
    }
    .model-parameters-row {
        margin: 0 0 8px 0;
    }
    .model-parameters-label,
    .node-parameters-label {
        flex: 0 0 96px;
        display: flex;
        align-items: center;
        gap: 6px;
        color: #475569;
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 0.05em;
        line-height: 1.25;
        text-transform: uppercase;
    }
    .parameters-label-icon {
        flex: 0 0 auto;
        width: 26px;
        height: 26px;
        display: inline-grid;
        place-items: center;
        border-radius: 50%;
        background: #edf5ff;
        color: #1769c2;
    }
    .parameters-label-icon svg {
        width: 15px;
        height: 15px;
        stroke: currentColor;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-width: 2;
    }
    .network-parameter {
        min-width: 0;
        padding: 11px 12px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 12px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .parameter-card-copy { min-width: 0; }
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
        .model-parameters-row,
        .node-parameters-panel { flex-direction: column; }
        .model-parameters-label,
        .node-parameters-label { flex-basis: auto; }
    }
    @media (max-width: 1400px) {
        .network-parameters,
        .node-parameters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .model-parameters-label,
        .node-parameters-label { flex-basis: 88px; }
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
        <div class="model-parameters-row" aria-label="Model parameters">
          <div class="model-parameters-label"><span class="parameters-label-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><circle cx="5" cy="5" r="2.5"></circle><circle cx="19" cy="8" r="2.5"></circle><circle cx="12" cy="19" r="2.5"></circle><path d="M7.2 6.2 16.7 7.3M6.6 7.2l4.2 9.4M17.7 10.1l-4.2 6.6"></path></svg></span><span>Model parameters</span></div>
          <div class="network-parameters">
            <div class="network-parameter"><span class="parameter-card-copy"><span class="network-parameter-label">Nodes</span><span class="network-parameter-value">{model_node_count}</span></span></div>
            <div class="network-parameter"><span class="parameter-card-copy"><span class="network-parameter-label">Edges</span><span class="network-parameter-value">{model_edge_count}</span></span></div>
            <div class="network-parameter"><span class="parameter-card-copy"><span class="network-parameter-label">Input nodes</span><span class="network-parameter-value">{model_input_node_count}</span></span></div>
            <div class="network-parameter"><span class="parameter-card-copy"><span class="network-parameter-label">Output nodes</span><span class="network-parameter-value">{model_output_node_count}</span></span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    node_parameters_slot = st.container(key="selected-node-parameters-slot")
    graph_click_event = render_clickable_network_graph(
        g,
        focused_graph_node_id,
        graph_focus_context,
    )
    if isinstance(graph_click_event, dict):
        click_event_id = str(graph_click_event.get("event_id", ""))
        clicked_node_id = str(graph_click_event.get("node_id", ""))
        clear_focus_requested = bool(graph_click_event.get("clear_focus"))
        clicked_node_name = graph_node_name_by_id.get(clicked_node_id)
        if click_event_id and click_event_id != st.session_state.get("_last_graph_click_event"):
            st.session_state["_last_graph_click_event"] = click_event_id
            if clear_focus_requested:
                clear_graph_focus()
            else:
                st.session_state["_graph_focus_context"] = graph_focus_context
                st.session_state["_graph_focus_node_id"] = clicked_node_id
            if not clear_focus_requested and clicked_node_name in node_names and clicked_node_name != selected_node_name:
                st.session_state["node_schemata_select"] = clicked_node_name
                selected_node_name = clicked_node_name

selected_node_name = node_selector_slot.selectbox(
    "Select node for F' / F'' & canalization map",
    node_names,
    key="node_schemata_select",
    on_change=focus_node_selected_in_sidebar,
)

st.sidebar.markdown("#### Upload your model")
uploaded_cnet = st.sidebar.file_uploader(
    "Upload a .cnet Boolean network file",
    type=["cnet", "txt"],
    key="uploaded_cnet_file",
    label_visibility="collapsed",
)
if uploaded_cnet is not None and not use_uploaded:
    st.rerun()
if uploaded_bn is not None:
    st.sidebar.success(f"Loaded uploaded network: {uploaded_name}")
elif uploaded_error:
    st.sidebar.error("Could not load the uploaded CNET file.")
    st.sidebar.caption(uploaded_error)

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
        display: flex;
        align-items: stretch;
        gap: 14px;
        margin: 0;
        padding: 0;
        border: 0;
        border-radius: 0;
        background: transparent;
    }}
    .st-key-selected-node-parameters-slot {{
        min-height: 142px;
    }}
    .node-parameters-label {{
        flex: 0 0 96px;
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 0;
        color: #475569;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}
    .node-parameters {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        flex: 1 1 auto;
        gap: 10px;
    }}
    .node-parameter {{
        min-width: 0;
        padding: 10px 12px;
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 10px;
        background: #ffffff;
        display: flex;
        align-items: center;
        gap: 9px;
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
        .node-parameters-panel {{ flex-direction: column; }}
        .node-parameters-label {{ flex-basis: auto; }}
        .node-parameters {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .st-key-selected-node-parameters-slot {{ min-height: 210px; }}
    }}
    @media (min-width: 651px) and (max-width: 1400px) {{
        .node-parameters {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .st-key-selected-node-parameters-slot {{ min-height: 164px; }}
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

with node_parameters_slot:
    st.markdown(
        f"""
        <div class="node-parameters-panel" aria-label="Selected node parameters">
          <div class="node-parameters-label"><span class="parameters-label-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M4 7h16M4 17h16M9 4v6M15 14v6"></path><circle cx="9" cy="7" r="2"></circle><circle cx="15" cy="17" r="2"></circle></svg></span><span>Selected node parameters</span></div>
          <div class="node-parameters">
            <div class="node-parameter"><span class="parameter-card-copy"><span class="node-parameter-label">Inputs</span><span class="node-parameter-value">{selected_node_input_count}</span></span></div>
            <div class="node-parameter"><span class="parameter-card-copy"><span class="node-parameter-label">Sensitivity</span><span class="node-parameter-value">{selected_node_sensitivity}</span></span></div>
            <div class="node-parameter"><span class="parameter-card-copy"><span class="node-parameter-label">Effective connectivity</span><span class="node-parameter-value">{selected_node_effective_connectivity}</span></span></div>
            <div class="node-parameter"><span class="parameter-card-copy"><span class="node-parameter-label">Bias</span><span class="node-parameter-value">{selected_node_bias}</span></span></div>
          </div>
        </div>
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

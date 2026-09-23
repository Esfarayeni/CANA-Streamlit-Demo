"""Graphviz-based network rendering and the accompanying Streamlit legend."""

from __future__ import annotations

import math
import re
from typing import Any
from pathlib import Path

import graphviz
import matplotlib as mpl
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import LinearSegmentedColormap

from constants import (
    ARROWSIZE,
    CANVAS_INCH,
    DEFAULT_OUTLINE,
    FONT_SIZE,
    ISOLATED_OUTLINE,
    NEG_EDGE_COLOR,
    NEG_EDGE_STYLE,
    NODE_PENWIDTH,
    PENWIDTH_MAX,
    POS_EDGE_COLOR,
    RADIUS,
    SPECIAL_OUTLINE,
    ZERO_EDGE_COLOR,
    ZERO_EDGE_STYLE,
)
from metrics import metric_to_width


NODE_COLOR_MAP = LinearSegmentedColormap.from_list("network_node_values", ["white", "#d62728"])
NODE_COLOR_MAP.set_under("#2ca02c")
_COMPASS = ["e", "ne", "n", "nw", "w", "sw", "s", "se"]

NETWORK_GRAPH_COMPONENT = components.declare_component(
    "network_graph_component_v4",
    path=str(Path(__file__).resolve().parent / "network_graph_component"),
)


@st.cache_data(show_spinner=False, max_entries=32)
def graphviz_svg_from_source(source: str, engine: str = "dot") -> str:
    """Render deterministic Graphviz source as a browser-safe SVG string."""
    svg = graphviz.Source(source, engine=engine).pipe(format="svg", quiet=True).decode("utf-8")
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg, flags=re.IGNORECASE)
    return re.sub(r"<!DOCTYPE[^>]*>", "", svg, flags=re.IGNORECASE).strip()


def render_clickable_network_graph(
    graph: graphviz.Digraph,
    focused_node_id: object,
    context_id: object,
    view_mode: str = "2D",
    sphere_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render either the original Graphviz view or the interactive spherical view."""
    svg = ""
    if view_mode == "2D":
        try:
            svg = graphviz_svg_from_source(graph.source, graph.engine)
        except graphviz.backend.execute.CalledProcessError:
            st.warning(
                "This model's network graph could not be rendered. Try another metric "
                "or model; the rest of the explorer remains available."
            )
            return {}
    return NETWORK_GRAPH_COMPONENT(
        svg=svg,
        view_mode=view_mode,
        sphere_data=sphere_data or {},
        focused_node_id=str(focused_node_id or ""),
        context_id=str(context_id or ""),
        key="network-graph-component",
        default={},
    )


def circular_positions(graph: Any, radius: float = RADIUS) -> tuple[list[Any], dict[Any, tuple[float, float]]]:
    """Place nodes in deterministic label order on a fixed circular layout."""
    nodes = list(graph.nodes())
    if not nodes:
        return [], {}
    sorted_nodes = sorted(nodes, key=lambda node_id: graph.nodes[node_id].get("label", str(node_id)))
    positions = {
        node_id: (
            radius * np.cos(2 * np.pi * index / len(sorted_nodes)),
            radius * np.sin(2 * np.pi * index / len(sorted_nodes)),
        )
        for index, node_id in enumerate(sorted_nodes)
    }
    return sorted_nodes, positions


def spherical_positions(graph: Any, radius: float = RADIUS) -> tuple[list[Any], dict[Any, tuple[float, float, float]]]:
    """Place nodes uniformly on a deterministic Fibonacci sphere."""
    nodes = list(graph.nodes())
    if not nodes:
        return [], {}
    sorted_nodes = sorted(nodes, key=lambda node_id: graph.nodes[node_id].get("label", str(node_id)))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    positions = {}
    for index, node_id in enumerate(sorted_nodes):
        z_position = 1.0 - (2.0 * (index + 0.5) / len(sorted_nodes))
        ring_radius = math.sqrt(max(0.0, 1.0 - z_position * z_position))
        angle = golden_angle * index
        positions[node_id] = (
            radius * ring_radius * math.cos(angle),
            radius * ring_radius * math.sin(angle),
            radius * z_position,
        )
    return sorted_nodes, positions


def _node_fill_color(value: float, maximum: float, zero_is_nonpositive: bool) -> str:
    zero_value = value <= 0 if zero_is_nonpositive else value == 0
    if zero_value:
        return "#2ca02c"
    normalizer = mpl.colors.Normalize(vmin=1e-16, vmax=maximum if maximum > 0 else 1.0)
    return mpl.colors.rgb2hex(NODE_COLOR_MAP(normalizer(value)))


def spherical_graph_data(
    source_graph: Any,
    node_values: dict[Any, float],
    special_nodes: set[Any],
    isolated_nodes: set[Any],
    metric: str,
    threshold: float,
    edge_values: dict[tuple[Any, Any], float] | None = None,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> dict[str, Any]:
    """Build JSON-safe node and visible-edge data for the canvas sphere renderer."""
    del minimum  # Kept for symmetry with the 2D metric-rendering inputs.
    _, positions = spherical_positions(source_graph)
    zero_is_nonpositive = metric != "Edge effectiveness"
    node_maximum = max(node_values.values()) if node_values else 1.0
    nodes = []
    for node_id, (x_position, y_position, z_position) in positions.items():
        value = float(node_values.get(node_id, 0.0))
        nodes.append({
            "id": str(node_id),
            "label": str(source_graph.nodes[node_id].get("label", node_id)),
            "x": x_position,
            "y": y_position,
            "z": z_position,
            "fill": _node_fill_color(value, node_maximum, zero_is_nonpositive),
            "outline": (
                ISOLATED_OUTLINE if node_id in isolated_nodes else
                SPECIAL_OUTLINE if node_id in special_nodes else DEFAULT_OUTLINE
            ),
        })

    edges = []
    for source, target, data in source_graph.edges(data=True):
        value = float(data.get("weight", 0.0)) if metric == "Edge effectiveness" else float((edge_values or {}).get((source, target), 0.0))
        compared_value = abs(value) if metric == "Correlation" else value
        is_zero = np.isclose(value, 0.0)
        show_zero = metric in {"Edge effectiveness", "Activity", "Correlation"} and np.isclose(threshold, 0.0) and is_zero
        if not show_zero and compared_value <= threshold:
            continue
        if show_zero:
            width, color, dashed = 3.5, ZERO_EDGE_COLOR, True
        elif metric == "Edge effectiveness":
            width = max(0.5, min(PENWIDTH_MAX, PENWIDTH_MAX * value))
            color, dashed = "#111827", False
        else:
            width = metric_to_width(abs(value) if metric == "Correlation" else value, 0.0, maximum)
            color, dashed = "#111827", metric == "Correlation" and value < 0
        edges.append({
            "source": str(source), "target": str(target), "width": float(width),
            "color": color, "dashed": dashed,
        })
    return {"nodes": nodes, "edges": edges}


def _graph_with_standard_attributes(node_width_in: float) -> graphviz.Digraph:
    graph = graphviz.Digraph(engine="neato")
    graph.attr("graph", size=f"{CANVAS_INCH},{CANVAS_INCH}!", ratio="1", margin="0.2", pad="0.1", splines="line")
    graph.attr(
        "node",
        pin="true",
        shape="circle",
        fixedsize="true",
        width=f"{node_width_in:.2f}",
        style="filled",
        fontname="Helvetica",
        fontsize=FONT_SIZE,
        penwidth=str(NODE_PENWIDTH),
    )
    graph.attr("edge", arrowhead="normal", arrowsize=ARROWSIZE, color="black")
    return graph


def _add_nodes(
    graphviz_graph: graphviz.Digraph,
    source_graph: Any,
    node_values: dict[Any, float],
    special_nodes: set[Any],
    positions: dict[Any, tuple[float, float]],
    isolated_nodes: set[Any],
    zero_is_nonpositive: bool,
) -> float:
    for node_id in source_graph.nodes():
        node_values.setdefault(node_id, 0.0)
    maximum = max(node_values.values()) if node_values else 1.0
    normalizer = mpl.colors.Normalize(vmin=1e-16, vmax=maximum if maximum > 0 else 1.0)
    for node_id, (x_position, y_position) in positions.items():
        value = node_values.get(node_id, 0.0)
        zero_value = value <= 0 if zero_is_nonpositive else value == 0
        fill = "#2ca02c" if zero_value else mpl.colors.rgb2hex(NODE_COLOR_MAP(normalizer(value)))
        outline = (
            ISOLATED_OUTLINE if node_id in isolated_nodes
            else SPECIAL_OUTLINE if node_id in special_nodes
            else DEFAULT_OUTLINE
        )
        graphviz_graph.node(
            str(node_id),
            source_graph.nodes[node_id].get("label", str(node_id)),
            pos=f"{x_position:.3f},{y_position:.3f}!",
            color=outline,
            fillcolor=fill,
        )
    return maximum


def build_graphviz_effective(
    effective_graph: Any,
    node_values: dict[Any, float],
    special_nodes: set[Any],
    positions: dict[Any, tuple[float, float]],
    node_width_in: float,
    isolated_nodes: set[Any] | None = None,
) -> tuple[graphviz.Digraph, float]:
    """Build the Graphviz nodes for an edge-effectiveness network view."""
    graphviz_graph = _graph_with_standard_attributes(node_width_in)
    maximum = _add_nodes(
        graphviz_graph, effective_graph, node_values, special_nodes, positions,
        isolated_nodes or set(), zero_is_nonpositive=False,
    )
    return graphviz_graph, maximum


def build_graphviz_structural(
    structural_graph: Any,
    node_values: dict[Any, float],
    special_nodes: set[Any],
    positions: dict[Any, tuple[float, float]],
    node_width_in: float,
    isolated_nodes: set[Any] | None = None,
) -> tuple[graphviz.Digraph, float]:
    """Build the Graphviz nodes for activity, excess, or correlation views."""
    graphviz_graph = _graph_with_standard_attributes(node_width_in)
    maximum = _add_nodes(
        graphviz_graph, structural_graph, node_values, special_nodes, positions,
        isolated_nodes or set(), zero_is_nonpositive=True,
    )
    return graphviz_graph, maximum


def _angle_to_compass(delta_x: float, delta_y: float) -> str:
    angle = math.degrees(math.atan2(delta_y, delta_x)) % 360.0
    return _COMPASS[int(round(angle / 45.0)) % len(_COMPASS)]


def curved_zero_edge_ports(source: Any, target: Any, positions: dict[Any, tuple[float, float]], should_curve: bool = False) -> dict[str, str]:
    """Offset only zero-value reciprocal edges so both directions remain visible."""
    if not should_curve or source not in positions or target not in positions:
        return {}
    source_x, source_y = positions[source]
    target_x, target_y = positions[target]
    delta_x, delta_y = target_x - source_x, target_y - source_y
    if np.isclose(delta_x, 0.0) and np.isclose(delta_y, 0.0):
        return {}
    tail_index = (_COMPASS.index(_angle_to_compass(delta_x, delta_y)) + 1) % len(_COMPASS)
    head_index = (_COMPASS.index(_angle_to_compass(-delta_x, -delta_y)) + 1) % len(_COMPASS)
    return {"tailport": _COMPASS[tail_index], "headport": _COMPASS[head_index]}


def _has_reciprocal_edge(graph: Any, source: Any, target: Any) -> bool:
    try:
        return graph.has_edge(target, source)
    except Exception:
        return False


def _add_zero_edge(graphviz_graph: graphviz.Digraph, graph: Any, source: Any, target: Any, positions: dict[Any, tuple[float, float]]) -> None:
    graphviz_graph.edge(
        str(source), str(target), penwidth="3.5", color=ZERO_EDGE_COLOR, style=ZERO_EDGE_STYLE,
        **curved_zero_edge_ports(source, target, positions, _has_reciprocal_edge(graph, source, target)),
    )


def add_edges_effective(graphviz_graph: graphviz.Digraph, effective_graph: Any, threshold: float, positions: dict[Any, tuple[float, float]]) -> None:
    """Render above-threshold effectiveness edges and visible zero-value edges."""
    for source, target, data in effective_graph.edges(data=True):
        weight = float(data.get("weight", 0.0))
        if np.isclose(threshold, 0.0) and np.isclose(weight, 0.0):
            _add_zero_edge(graphviz_graph, effective_graph, source, target, positions)
        elif weight > threshold:
            width = max(0.5, min(PENWIDTH_MAX, PENWIDTH_MAX * weight)) if weight > 0 else 1.5
            graphviz_graph.edge(str(source), str(target), penwidth=f"{width:.2f}", color="black")


def add_edges_structural(
    graphviz_graph: graphviz.Digraph,
    structural_graph: Any,
    edge_values: dict[tuple[Any, Any], float],
    minimum: float,
    maximum: float,
    positions: dict[Any, tuple[float, float]],
    threshold: float | None = None,
    show_zero_dashed_at_zero_threshold: bool = False,
    signed_color: bool = False,
    threshold_on_abs: bool = False,
    negative_dashed: bool = False,
) -> None:
    """Render structural-edge metrics while retaining sign/zero conventions."""
    for source, target in structural_graph.edges():
        if (source, target) not in edge_values:
            continue
        value = float(edge_values[(source, target)])
        compared_value = abs(value) if threshold_on_abs else value
        if show_zero_dashed_at_zero_threshold and threshold is not None and np.isclose(threshold, 0.0) and np.isclose(value, 0.0):
            _add_zero_edge(graphviz_graph, structural_graph, source, target, positions)
            continue
        if threshold is not None and compared_value <= threshold:
            continue
        width = metric_to_width(abs(value) if threshold_on_abs else value, minimum, maximum)
        color, style = "black", "solid"
        if signed_color:
            color = NEG_EDGE_COLOR if value < 0 else POS_EDGE_COLOR
            style = NEG_EDGE_STYLE if value < 0 and negative_dashed else "solid"
        graphviz_graph.edge(str(source), str(target), penwidth=f"{width:.2f}", color=color, style=style)


def _legend_line(color: str = "black", style: str = "solid", width: int = 3) -> str:
    return (
        f"<span style='display:inline-block; width:36px; vertical-align:middle; "
        f"border-top:{width}px {'solid' if style == 'solid' else 'dashed'} {color}; margin-right:8px;'></span>"
    )


def _legend_node(fill: str = "#ffffff", outline: str = "#000000") -> str:
    return (
        f"<span style='display:inline-block; width:16px; height:16px; border-radius:50%; "
        f"background:{fill}; border:3px solid {outline}; margin-right:8px; vertical-align:middle;'></span>"
    )


def render_graph_legend(metric: str, threshold_value: float, degree_mode: str, scale_max: float) -> None:
    """Render the context-sensitive node, edge, and color-scale legend."""
    del threshold_value  # retained for a stable UI-facing function signature
    node_items = [
        (_legend_node(fill="#f7f7f7", outline=DEFAULT_OUTLINE), "regular node"),
        (_legend_node(fill="#f7f7f7", outline=SPECIAL_OUTLINE), "input node"),
        (_legend_node(fill="#f7f7f7", outline=ISOLATED_OUTLINE), "isolated after thresholding"),
    ]
    edge_items = {
        "Edge effectiveness": [(_legend_line(width=3), "thicker = larger effectiveness"), (_legend_line("red", "dashed", 3), "weight = 0")],
        "Activity": [(_legend_line(width=3), "thicker = larger activity"), (_legend_line("red", "dashed", 3), "activity = 0")],
        "Excess canalization": [(_legend_line(width=3), "thicker = larger excess canalization")],
    }.get(metric, [
        (_legend_line(width=3), "positive correlation; thicker = larger |correlation|"),
        (_legend_line("black", "dashed", 3), "negative correlation"),
        (_legend_line("red", "dashed", 3), "correlation = 0"),
    ])
    row_style = "margin:6px 0; padding:6px 8px; border-radius:10px; background:rgba(248,250,252,0.95); border:1px solid rgba(148,163,184,0.14);"
    rows = lambda items: "".join(f"<div style='{row_style}'>{icon}<span>{label}</span></div>" for icon, label in items)
    maximum = float(scale_max) if float(scale_max) > 0 else 0.0
    st.markdown(f"""
    <div style="font-weight:700; font-size:0.88rem; letter-spacing:0.02em; margin-bottom:6px; color:#0f172a;">Nodes</div>
    {rows(node_items)}
    <div style="font-weight:700; font-size:0.88rem; letter-spacing:0.02em; margin:12px 0 6px; color:#0f172a;">Edges</div>
    {rows(edge_items)}
    <div style="font-weight:700; font-size:0.88rem; letter-spacing:0.02em; margin:12px 0 6px; color:#0f172a;">Node color scale ({degree_mode.lower()})</div>
    <div style="margin-top:6px; padding:8px 10px; border-radius:12px; background:rgba(248,250,252,0.95); border:1px solid rgba(148,163,184,0.14);">
      <div style="width:100%; height:16px; border:1px solid #999; border-radius:8px; background:linear-gradient(to right,#2ca02c 0%,#f7f7f7 18%,#fddbc7 45%,#fca082 70%,#f1695c 85%,#d62728 100%);"></div>
      <div style="display:flex; justify-content:space-between; font-size:0.9em; margin-top:4px;"><span>0.00</span><span>{maximum:.2f}</span></div>
      <div style="font-size:0.9em; color:#444; margin-top:4px;">Green = zero, darker red = larger {degree_mode.lower()} value</div>
    </div>
    """, unsafe_allow_html=True)


def clean_svg_for_panel(svg_text: str) -> str:
    """Remove fixed SVG dimensions so schemata and maps scale inside their panels."""
    svg_text = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>|</?div[^>]*>", "", svg_text, flags=re.IGNORECASE).strip()
    svg_text = re.sub(r'\swidth="[^"]*"', "", svg_text, count=1)
    svg_text = re.sub(r'\sheight="[^"]*"', "", svg_text, count=1)
    if "preserveAspectRatio=" in svg_text:
        svg_text = re.sub(r'preserveAspectRatio="[^"]*"', 'preserveAspectRatio="xMidYMid meet"', svg_text, count=1)
    else:
        svg_text = re.sub(r"<svg\b", '<svg preserveAspectRatio="xMidYMid meet"', svg_text, count=1)
    if 'class="canalization-map-svg"' not in svg_text:
        svg_text = re.sub(r"<svg\b", '<svg class="canalization-map-svg"', svg_text, count=1)
    return svg_text

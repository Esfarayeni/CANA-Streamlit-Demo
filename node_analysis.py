"""Node-level CANA analysis: parameter formatting, schemata, and maps."""

from __future__ import annotations

import base64
import io
from copy import copy
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib.text import Text

from cana.drawing.canalizing_map import draw_canalizing_map_graphviz
from graph_rendering import clean_svg_for_panel, graphviz_svg_from_source


def format_node_parameter(metric_fn: Callable[[], Any]) -> str:
    """Return a consistently formatted node metric, or an em dash if absent."""
    try:
        value = float(metric_fn())
        return f"{value:.2f}" if np.isfinite(value) else "—"
    except Exception:
        return "—"


def _safe_compute_schemata(node: Any) -> None:
    """Populate CANA's optional prime-implicant and two-symbol caches."""
    for keyword in ("prime_implicants", "two_symbols"):
        try:
            node._check_compute_canalization_variables(**{keyword: True})
        except Exception:
            pass
    if getattr(node, "_prime_implicants", None) is None or getattr(node, "_two_symbols", None) is None:
        try:
            node.schemata()
        except Exception:
            pass


def _value_style(value: str, two_symbol: bool = False) -> tuple[str, str, str]:
    if value == "1":
        return "black", "white", value
    if value in {"#", "2"}:
        return "#cccccc", "black", "#" if two_symbol else value
    return "white", "black", value


def _hide_axis(axis: Any) -> None:
    axis.xaxis.tick_top()
    axis.tick_params(which="major", pad=7)
    for tick in [*axis.xaxis.get_major_ticks(), *axis.yaxis.get_major_ticks()]:
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_schemata(node: Any):
    """Build the prime-implicant and two-symbol schemata figure for a node."""
    _safe_compute_schemata(node)
    input_count = node.k if node.k >= 1 else 1
    inputs = node.inputs if not node.constant else [node.name]
    input_labels = [node.network.get_node_name(value)[0] if node.network is not None else value for value in inputs]
    prime_implicants = getattr(node, "_prime_implicants", {}) or {}
    prime_zero, prime_one = prime_implicants.get("0", []) or [], prime_implicants.get("1", []) or []
    two_symbols = getattr(node, "_two_symbols", None) or [[], []]
    two_zero = two_symbols[0] if len(two_symbols) > 0 and two_symbols[0] is not None else []
    two_one = two_symbols[1] if len(two_symbols) > 1 and two_symbols[1] is not None else []
    prime_count, two_count = len(prime_zero) + len(prime_one), len(two_zero) + len(two_one)

    cell_width, cell_gap, row_gap, separator, panel_gap, dpi = 60.0, 0, 6, 15, 60, 150.0
    top, right, bottom, left = 160, 25, 25, 60
    panel_width = input_count * (cell_width + cell_gap) + 21 + cell_width
    prime_height = max(prime_count, 1) * (cell_width + row_gap) + separator - row_gap
    two_height = max(two_count, 1) * (cell_width + row_gap) + separator - row_gap
    figure_width = left + panel_width + panel_gap + panel_width + right
    figure_height = bottom + max(prime_height, two_height) + top
    figure = plt.figure(figsize=(figure_width / dpi, figure_height / dpi), facecolor="w", dpi=dpi)
    first_axis = figure.add_axes((left / figure_width, bottom / figure_height, panel_width / figure_width, prime_height / figure_height), aspect=1, label="PI")
    second_axis = figure.add_axes(((left + panel_width + panel_gap) / figure_width, bottom / figure_height, panel_width / figure_width, two_height / figure_height), aspect=1, label="TS")

    _draw_prime_implicants(first_axis, prime_one, prime_zero, input_labels, node.name, cell_width, cell_gap, row_gap, separator, panel_width, prime_height)
    _draw_two_symbols(second_axis, two_one, two_zero, input_labels, node.name, cell_width, cell_gap, row_gap, separator, panel_width, two_height)
    return figure


def _draw_prime_implicants(axis: Any, output_one: list[Any], output_zero: list[Any], input_labels: list[str], node_name: str, cell_width: float, cell_gap: float, row_gap: float, separator: float, panel_width: float, panel_height: float) -> None:
    y_position, labels, patches, tick_positions = 0.0, [], [], []
    for output, schemata in ((1, output_one), (0, output_zero)):
        for schema in schemata:
            x_position, tick_positions = 0.0, []
            for value in schema:
                fill, text_color, text = _value_style(value)
                axis.add_artist(Text(x_position + cell_width / 2, y_position + cell_width * 0.4, text=text, color=text_color, va="center", ha="center", fontsize=14, family="serif"))
                patches.append(Rectangle((x_position, y_position), cell_width, cell_width, facecolor=fill, edgecolor="black"))
                tick_positions.append(x_position + cell_width / 2)
                x_position += cell_width + cell_gap
            x_position += 21
            patches.append(Rectangle((x_position, y_position), cell_width, cell_width, facecolor="black" if output else "white", edgecolor="black"))
            axis.add_artist(Text(x_position - 10.5, y_position + cell_width * 0.4, text=":", color="black", va="center", ha="center", fontsize=14, weight="bold", family="serif"))
            axis.add_artist(Text(x_position + cell_width / 2, y_position + cell_width * 0.4, text=str(output), color="white" if output else "black", va="center", ha="center", fontsize=14, family="serif"))
            tick_positions.append(x_position + cell_width / 2)
            labels.append(y_position + cell_width / 2)
            y_position += cell_width + row_gap
        y_position += separator
    if patches:
        axis.add_collection(PatchCollection(patches, match_original=True))
    axis.set_yticks(labels)
    axis.set_yticklabels([r"$f^{'}_{%d}$" % index for index in range(len(labels), 0, -1)], fontsize=14)
    axis.set_xticks(tick_positions if tick_positions else [])
    if tick_positions:
        axis.set_xticklabels(input_labels + [str(node_name)], rotation=90, fontsize=14)
    _hide_axis(axis)
    axis.set_xlim(-1, panel_width + 1)
    axis.set_ylim(-1, panel_height + 1)


def _draw_two_symbols(axis: Any, output_one: list[Any], output_zero: list[Any], input_labels: list[str], node_name: str, cell_width: float, cell_gap: float, row_gap: float, separator: float, panel_width: float, panel_height: float) -> None:
    y_position, labels, boxes, symbols, tick_positions = 0.0, [], [], [], []
    symbol_templates = [
        Circle((0, 0), radius=5, facecolor="white", edgecolor="black"),
        RegularPolygon((0, 0), numVertices=3, radius=5, orientation=0, facecolor="white", edgecolor="black"),
    ]
    for output, schemata in ((1, output_one), (0, output_zero)):
        for schema, prime_sets, symbol_sets in schemata:
            del symbol_sets  # CANA encodes the displayed symbols in prime_sets.
            x_position, tick_positions = 0.0, []
            for input_index, value in enumerate(schema):
                fill, text_color, text = _value_style(value, two_symbol=True)
                relevant_sets = [index for index, item in enumerate(prime_sets) if input_index in item] if len(prime_sets) else []
                for position, set_index in enumerate(relevant_sets, start=1):
                    if set_index >= len(symbol_templates):
                        continue
                    marker = copy(symbol_templates[set_index])
                    marker.set_facecolor("none")
                    marker.center = (np.linspace(x_position, x_position + cell_width, len(relevant_sets) + 2)[position], y_position + cell_width * 0.8)
                    marker.set_zorder(10)
                    marker.set_edgecolor("#a6a6a6" if value == "1" else "black")
                    symbols.append(marker)
                    axis.add_patch(marker)
                axis.add_artist(Text(x_position + cell_width / 2, y_position + cell_width * 0.4, text=text, color=text_color, va="center", ha="center", fontsize=14, family="serif"))
                boxes.append(Rectangle((x_position, y_position), cell_width, cell_width, facecolor=fill, edgecolor="#4c4c4c", zorder=2))
                tick_positions.append(x_position + cell_width / 2)
                x_position += cell_width + cell_gap
            x_position += 21
            boxes.append(Rectangle((x_position, y_position), cell_width, cell_width, facecolor="black" if output else "white", edgecolor="#4c4c4c"))
            axis.add_artist(Text(x_position - 10.5, y_position + cell_width / 2, text=":", color="black", va="center", ha="center", fontsize=14, weight="bold", family="serif"))
            axis.add_artist(Text(x_position + cell_width / 2, y_position + cell_width * 0.4, text=str(output), color="white" if output else "black", va="center", ha="center", fontsize=14, family="serif"))
            tick_positions.append(x_position + cell_width / 2)
            labels.append(y_position + cell_width / 2)
            y_position += cell_width + row_gap
        y_position += separator
    if boxes:
        axis.add_collection(PatchCollection(boxes, match_original=True))
    if symbols:
        axis.add_collection(PatchCollection(symbols, match_original=True))
    axis.set_yticks(labels)
    axis.set_yticklabels([r"$f^{''}_{%d}$" % index for index in range(len(labels), 0, -1)], fontsize=14)
    axis.set_xticks(tick_positions if tick_positions else [])
    if tick_positions:
        axis.set_xticklabels(input_labels + [str(node_name)], rotation=90, fontsize=14)
    _hide_axis(axis)
    axis.set_xlim(-1, panel_width + 1)
    axis.set_ylim(-1, panel_height + 1)


def render_schemata_png(node: Any) -> str:
    """Render a node's schemata as a base64 PNG for stable Streamlit panels."""
    figure = plot_schemata(node)
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.35, dpi=200, facecolor="white")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    finally:
        plt.close(figure)


def render_canalization_map_svg(node: Any) -> tuple[str, int]:
    """Render a node's upper-bound canalization map as scalable SVG payload."""
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

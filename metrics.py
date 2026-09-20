"""Cached-ready numeric and graph metrics for Boolean-network exploration.

The functions here are deliberately independent of Streamlit and presentation
code, which keeps numerical behavior straightforward to test.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from constants import MAX_CORRELATION_INPUTS, MAX_WIDTH, MIN_WIDTH


def normalize_name(value: object) -> str:
    """Return a stable, case-insensitive label for node matching."""
    return str(value).strip().lower()


def detect_special_nodes(bn: Any, graph: Any) -> set[Any]:
    """Find input and self-regulated-only nodes in a CANA graph."""
    special_labels: set[str] = set()
    for node in bn.nodes:
        name = getattr(node, "name", f"node_{getattr(node, 'id', 'X')}")
        inputs = list(getattr(node, "inputs", []) or [])
        input_count = getattr(node, "k", len(inputs))

        def is_self(input_node: Any) -> bool:
            node_id = getattr(node, "id", None)
            input_id = getattr(input_node, "id", None)
            if node_id is not None and input_id is not None:
                return input_id == node_id
            return normalize_name(getattr(input_node, "name", input_node)) == normalize_name(name)

        if input_count == 0 or (input_count == 1 and len(inputs) == 1 and is_self(inputs[0])):
            special_labels.add(normalize_name(name))

    labels = {
        node_id: normalize_name(graph.nodes[node_id].get("label", node_id))
        for node_id in graph.nodes()
    }
    special_nodes = {node_id for node_id, label in labels.items() if label in special_labels}
    if special_nodes:
        return special_nodes

    # Some imported formats do not retain CANA node labels.  Use the graph
    # topology as a reliable fallback for input/self-loop-only nodes.
    return {
        node_id for node_id in graph.nodes()
        if graph.in_degree(node_id) == 0
        or (
            graph.in_degree(node_id) == 1
            and graph.has_edge(node_id, node_id)
            and all(parent == node_id for parent in graph.predecessors(node_id))
        )
    }


def threshold_graph(effective_graph: Any, threshold: float) -> Any:
    """Return an effective graph containing strictly above-threshold edges."""
    filtered_graph = effective_graph.copy()
    filtered_graph.remove_edges_from([
        (source, target)
        for source, target, data in filtered_graph.edges(data=True)
        if float(data.get("weight", 0.0)) <= threshold
    ])
    return filtered_graph


def isolated_nodes_after_threshold(
    graph: Any,
    edge_values: dict[tuple[Any, Any], float] | None = None,
    threshold: float | None = None,
    use_absolute_values: bool = False,
) -> set[Any]:
    """Find nodes with no remaining incident edges after filtering."""
    filtered_graph = graph.copy()
    if edge_values is not None and threshold is not None:
        filtered_graph.remove_edges_from([
            (source, target)
            for source, target in filtered_graph.edges()
            if (source, target) in edge_values
            and (abs(edge_values[(source, target)]) if use_absolute_values else edge_values[(source, target)])
            <= threshold
        ])
    return {
        node_id for node_id in filtered_graph.nodes()
        if filtered_graph.in_degree(node_id) == 0 and filtered_graph.out_degree(node_id) == 0
    }


def node_values_from_thresholded_effective(graph: Any, degree_mode: str = "Out-degree") -> dict[Any, float]:
    """Sum retained effective-edge weights for each node."""
    if degree_mode == "In-degree":
        return {node_id: graph.in_degree(node_id, weight="weight") for node_id in graph.nodes()}
    return {node_id: graph.out_degree(node_id, weight="weight") for node_id in graph.nodes()}


def node_values_from_thresholded_structural(
    graph: Any,
    edge_values: dict[tuple[Any, Any], float],
    threshold: float,
    degree_mode: str = "Out-degree",
    use_absolute_values: bool = False,
) -> dict[Any, float]:
    """Accumulate metric values over edges that survive a threshold."""
    node_values = {node_id: 0.0 for node_id in graph.nodes()}
    for (source, target), value in edge_values.items():
        comparison_value = abs(value) if use_absolute_values else value
        if comparison_value <= threshold:
            continue
        contribution = abs(value) if use_absolute_values else max(0.0, value)
        node_values[target if degree_mode == "In-degree" else source] += contribution
    return node_values


def metric_to_width(value: float, minimum: float, maximum: float, minimum_width: float = MIN_WIDTH, maximum_width: float = MAX_WIDTH) -> float:
    """Map a metric value to the bounded edge-width range used in the UI."""
    fraction = (value - minimum) / (maximum - minimum) if maximum > minimum else 0.5
    return minimum_width + fraction * (maximum_width - minimum_width)


def _graph_node_for_reference(reference: Any, graph: Any, nodes_by_label: dict[str, Any]) -> Any | None:
    """Resolve a CANA input reference to a node identifier in ``graph``."""
    for candidate in (reference, getattr(reference, "id", None)):
        if candidate is None:
            continue
        try:
            if candidate in graph:
                return candidate
        except TypeError:
            continue
    for name in (getattr(reference, "name", None), str(reference)):
        if name is not None and (match := nodes_by_label.get(normalize_name(name))) is not None:
            return match
    return None


def ordered_cana_edges(node: Any, graph: Any) -> list[tuple[Any, Any] | None]:
    """Map CANA's declared input order to incoming graph edges.

    Activities and effectiveness values are positional, so preserving this
    order is essential: graph predecessor iteration alone can be incorrect.
    """
    nodes_by_label = {
        normalize_name(graph.nodes[node_id].get("label", node_id)): node_id
        for node_id in graph.nodes()
    }
    target = _graph_node_for_reference(node, graph, nodes_by_label)
    if target is None:
        return []
    return [
        (source, target) if source is not None and graph.has_edge(source, target) else None
        for input_reference in list(getattr(node, "inputs", []) or [])
        for source in [_graph_node_for_reference(input_reference, graph, nodes_by_label)]
    ]


def compute_structural_metrics(bn: Any, include_excess: bool = True):
    """Compute CANA activity and excess-canalization values keyed by edges."""
    structural_graph = bn.structural_graph()
    if structural_graph.number_of_nodes() == 0:
        return structural_graph, {}, (0.0, 1.0), {}, (0.0, 1.0)

    edge_activity: dict[tuple[Any, Any], float] = {}
    edge_excess: dict[tuple[Any, Any], float] = {}
    activity_values: list[float] = []
    excess_values: list[float] = []
    for node in bn.nodes:
        try:
            raw_activities = node.activities()
            activities = list(raw_activities) if raw_activities is not None else []
            raw_effectiveness = node.edge_effectiveness() if include_excess else None
            effectiveness = list(raw_effectiveness) if raw_effectiveness is not None else None
        except Exception:
            continue
        edges = ordered_cana_edges(node, structural_graph)
        for index, edge in enumerate(edges[:len(activities)]):
            if edge is None or not np.isfinite(activities[index]):
                continue
            activity = float(activities[index])
            edge_activity[edge] = activity
            activity_values.append(activity)
            if effectiveness is not None and index < len(effectiveness) and np.isfinite(effectiveness[index]):
                excess = float(effectiveness[index] - activity)
                edge_excess[edge] = excess
                excess_values.append(excess)

    activity_bounds = (float(np.min(activity_values)), float(np.max(activity_values))) if activity_values else (0.0, 1.0)
    excess_bounds = (float(np.min(excess_values)), float(np.max(excess_values))) if excess_values else (0.0, 1.0)
    return structural_graph, edge_activity, activity_bounds, edge_excess, excess_bounds


def safe_binary_corr(first: Any, second: Any) -> float:
    """Return a finite Pearson correlation, treating constant vectors as zero."""
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if first_values.size == 0 or first_values.size != second_values.size:
        return 0.0
    if np.allclose(first_values, first_values[0]) or np.allclose(second_values, second_values[0]):
        return 0.0
    correlation = np.corrcoef(first_values, second_values)[0, 1]
    return float(correlation) if np.isfinite(correlation) else 0.0


def compute_correlation_metrics(bn: Any):
    """Correlate each truth-table input column with its node's output.

    The input limit prevents materializing impractically large truth tables.
    The UI catches the resulting ValueError and presents it as a friendly
    model-specific error.
    """
    structural_graph = bn.structural_graph()
    if structural_graph.number_of_nodes() == 0:
        return structural_graph, {}, (0.0, 1.0)

    edge_correlation: dict[tuple[Any, Any], float] = {}
    absolute_values: list[float] = []
    for node in bn.nodes:
        inputs = list(getattr(node, "inputs", []) or [])
        input_count = getattr(node, "k", len(inputs))
        if input_count <= 0 or not inputs:
            continue
        if input_count > MAX_CORRELATION_INPUTS:
            name = getattr(node, "name", f"node_{getattr(node, 'id', 'X')}")
            raise ValueError(
                f"Correlation is limited to {MAX_CORRELATION_INPUTS} inputs per node; {name} has {input_count}."
            )
        raw_outputs = getattr(node, "outputs", None)
        outputs = np.asarray(list(raw_outputs) if raw_outputs is not None else [], dtype=float)
        if outputs.size != 2 ** input_count:
            continue
        row_ids = np.arange(outputs.size, dtype=np.uint64)
        for index, edge in enumerate(ordered_cana_edges(node, structural_graph)[:input_count]):
            if edge is None:
                continue
            input_column = ((row_ids >> np.uint64(input_count - index - 1)) & np.uint64(1)).astype(float)
            correlation = safe_binary_corr(input_column, outputs)
            edge_correlation[edge] = correlation
            absolute_values.append(abs(correlation))
    bounds = (float(np.min(absolute_values)), float(np.max(absolute_values))) if absolute_values else (0.0, 1.0)
    return structural_graph, edge_correlation, bounds

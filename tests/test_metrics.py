from types import SimpleNamespace

import networkx as nx
import numpy as np

from metrics import (
    compute_correlation_metrics,
    isolated_nodes_after_threshold,
    node_values_from_thresholded_structural,
    ordered_cana_edges,
    safe_binary_corr,
    threshold_graph,
)


def test_threshold_graph_keeps_only_strictly_above_threshold_edges():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=0.2)
    graph.add_edge("b", "c", weight=0.8)

    filtered = threshold_graph(graph, 0.2)

    assert list(filtered.edges()) == [("b", "c")]


def test_node_values_and_isolation_respect_absolute_thresholds():
    graph = nx.DiGraph([("a", "b"), ("b", "c")])
    values = {("a", "b"): -0.5, ("b", "c"): 0.7}

    assert node_values_from_thresholded_structural(
        graph, values, 0.6, use_absolute_values=True
    ) == {"a": 0.0, "b": 0.7, "c": 0.0}
    assert isolated_nodes_after_threshold(
        graph, values, 0.6, use_absolute_values=True
    ) == {"a"}


def test_ordered_cana_edges_uses_declared_input_order_not_graph_iteration_order():
    graph = nx.DiGraph()
    graph.add_nodes_from([(1, {"label": "target"}), (2, {"label": "first"}), (3, {"label": "second"})])
    graph.add_edge(3, 1)
    graph.add_edge(2, 1)
    node = SimpleNamespace(name="target", id=1, inputs=[SimpleNamespace(name="first"), SimpleNamespace(name="second")])

    assert ordered_cana_edges(node, graph) == [(2, 1), (3, 1)]


def test_safe_binary_corr_and_correlation_metrics_handle_constant_outputs():
    assert safe_binary_corr([0, 0], [1, 1]) == 0.0
    assert np.isclose(safe_binary_corr([0, 1], [0, 1]), 1.0)

    class FakeNetwork:
        def structural_graph(self):
            graph = nx.DiGraph()
            graph.add_nodes_from([(1, {"label": "target"}), (2, {"label": "input"})])
            graph.add_edge(2, 1)
            return graph

    node = SimpleNamespace(
        name="target", id=1, k=1, inputs=[SimpleNamespace(name="input")], outputs=[1, 1]
    )
    _, edge_values, bounds = compute_correlation_metrics(SimpleNamespace(nodes=[node], structural_graph=FakeNetwork().structural_graph))

    assert edge_values == {(2, 1): 0.0}
    assert bounds == (0.0, 0.0)

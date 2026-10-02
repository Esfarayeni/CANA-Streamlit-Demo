import json

import networkx as nx

from graph_rendering import spherical_graph_data


def test_sphere_accessible_data_contains_node_types_and_numeric_values():
    graph = nx.DiGraph()
    graph.add_nodes_from([(1, {"label": "Input"}), (2, {"label": "Output"}), (3, {"label": "Isolated"})])
    graph.add_edge(1, 2, weight=0.75)
    data = spherical_graph_data(graph, {1: 0.75, 2: 0.0}, {1}, {3}, "Edge effectiveness", 0.0)

    nodes = {node["id"]: node for node in data["nodes"]}
    assert nodes["1"]["type"] == "Input node"
    assert nodes["2"]["type"] == "Regular node"
    assert nodes["3"]["type"] == "Isolated after thresholding"
    assert nodes["1"]["value"] == 0.75
    assert data["edges"][0]["value"] == 0.75
    json.dumps(data, allow_nan=False)


def test_sphere_accessible_edges_preserve_signed_metric_and_threshold():
    graph = nx.DiGraph([(1, 2), (2, 3)])
    data = spherical_graph_data(graph, {}, set(), set(), "Correlation", 0.3,
                                edge_values={(1, 2): -0.8, (2, 3): 0.2})

    assert len(data["edges"]) == 1
    assert data["edges"][0]["value"] == -0.8
    assert data["edges"][0]["dashed"] is True

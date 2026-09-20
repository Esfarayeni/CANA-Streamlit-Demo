# Canalization Explorer

An interactive Streamlit dashboard for exploring Boolean-network structure, logic, and canalization.

![CASCI Canalization Explorer logo](assets/casci_canalization_explorer_logo_transparent.png)

[GitHub repository](https://github.com/Esfarayeni/CANA-Streamlit-Demo) · [Python 3.11+](https://www.python.org/) · [Built with Streamlit](https://streamlit.io/) · [Powered by CANA](https://github.com/CASCI-lab/CANA)

Canalization Explorer lets researchers inspect Boolean-network models from the [Cell Collective](https://cellcollective.org/) and their own `.cnet` files. It combines network-level metrics with node-level Boolean schemata and canalization maps, making the relationship between a model’s topology and logic easier to explore.

Developed at the [CASCI Lab](https://casci.binghamton.edu/casci.php) and built with the [CANA Python library](https://github.com/CASCI-lab/CANA).

## Highlights

- Explore bundled Cell Collective Boolean-network models or upload a `.cnet` model.
- Compare edge effectiveness, activity, excess canalization, and Boolean input-output correlation.
- Filter edges by a threshold and inspect isolated nodes after filtering.
- Click a node to focus its incoming and outgoing connections; select a node from the sidebar to inspect its Boolean schemata and canalization map.
- View model size, input/output nodes, node sensitivity, effective connectivity, and bias.
- Access paper metadata and primary-reference links for bundled Cell Collective models.

## What is canalization?

Canalization describes how strongly particular input states constrain the output of a Boolean update rule. A highly canalizing input can determine a node's next state despite variation in other inputs. This explorer uses CANA to calculate and visualize these logical properties through prime implicants, two-symbol schemata, effective connectivity, and canalization maps.

## Quick start

### Prerequisites

- Python 3.11 or newer
- [Graphviz](https://graphviz.org/download/) installed and available on your `PATH` (used to render network and canalization-map SVGs)

### Install and run

```bash
git clone https://github.com/Esfarayeni/CANA-Streamlit-Demo.git
cd CANA-Streamlit-Demo

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run Main.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Using the explorer

1. Select a bundled model from **Select model**, or upload a `.cnet` file in **Upload your model**.
2. Choose an analysis metric and adjust the **Threshold** slider. Only values strictly above the threshold remain in the metric-focused view.
3. Click a graph node to focus it and highlight its regulators and targets. Click the graph background to return to the full network.
4. Use **Select node for F′ / F″ & canalization map** to view a node's prime-implicant schemata, two-symbol schemata, and upper-bound canalization map.

### Metrics

| Metric | Interpretation |
| --- | --- |
| **Edge effectiveness** | Influence of an input on a node's Boolean update rule. |
| **Activity** | Frequency with which an input changes the node output across the rule's truth table. |
| **Excess canalization** | Difference between effectiveness and activity, capturing redundancy/canalizing structure beyond direct activity. |
| **Correlation** | Pearson correlation between an input's truth-table column and the node output; negative values are rendered as dashed edges. |

## Project structure

| Path | Responsibility |
| --- | --- |
| [Main.py](Main.py) | Streamlit page layout, state, controls, and composition. |
| [constants.py](constants.py) | Application configuration, rendering defaults, and interactive limits. |
| [model_data.py](model_data.py) | Bundled model loading, `.cnet` uploads, validation, and Cell Collective metadata. |
| [metrics.py](metrics.py) | Edge/node metrics, thresholding, and network-derived values. |
| [graph_rendering.py](graph_rendering.py) | Graphviz SVG rendering, graph interaction component, and legends. |
| [node_analysis.py](node_analysis.py) | Schemata plots, canalization maps, and node-parameter formatting. |
| [ui_components.py](ui_components.py) | Reusable Streamlit presentation components. |
| [tests/](tests) | Tests for extracted model and metric logic. |

## Testing

Run the automated checks with:

```bash
pytest -q
```

The tests cover threshold semantics, metric aggregation, input-order mapping, safe correlation behavior, citation formatting, and upload validation.

## Data provenance and model metadata

Bundled models are provided by CANA's Cell Collective dataset. The checked-in [model_metadata.json](model_metadata.json) catalogue contains public Cell Collective model metadata and paper references. The app reads that local catalogue first so normal model selection does not depend on a network request; the public Cell Collective API is used only as a fallback for a missing record.

Refresh the catalogue deliberately after model updates:

```bash
python scripts/refresh_model_metadata.py
```

The refresh script queries public Cell Collective endpoints and updates `model_metadata.json`. Review the resulting diff before committing it.

## Limits and behavior

To keep interactive analysis responsive, uploaded models are limited to 5 MB, 500 nodes, 5,000 edges, and 18 inputs per node. Correlation analysis is limited to nodes with at most 16 inputs because a Boolean truth table grows exponentially with input count.

## Acknowledgments

- [CASCI Lab, Binghamton University](https://casci.binghamton.edu/casci.php)
- [CANA: Canalization Analysis](https://github.com/CASCI-lab/CANA)
- [Cell Collective](https://cellcollective.org/)
- [Graphviz](https://graphviz.org/)

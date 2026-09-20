from types import SimpleNamespace

import networkx as nx
import pytest

from model_data import format_primary_citation, get_bn_display_name, validate_uploaded_network


def test_display_name_uses_fallback_for_blank_names():
    assert get_bn_display_name(SimpleNamespace(name="  "), "Fallback") == "Fallback"
    assert get_bn_display_name(SimpleNamespace(name="Example"), "Fallback") == "Example"


def test_format_primary_citation_escapes_html_and_highlights_title():
    rendered = format_primary_citation("Author. <Title & study>. Journal.", "<Title & study>")

    assert "&lt;Title &amp; study&gt;" in rendered
    assert 'class="primary-paper-title"' in rendered


def test_validate_upload_rejects_empty_network():
    network = SimpleNamespace(nodes=[], structural_graph=lambda: nx.DiGraph())

    with pytest.raises(ValueError, match="does not contain any nodes"):
        validate_uploaded_network(network)

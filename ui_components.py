"""Small reusable Streamlit presentation components."""

import streamlit as st


def render_quick_guide() -> None:
    """Render the sidebar introduction and attribution."""
    with st.sidebar.expander("Quick guide", expanded=False):
        st.markdown(
            """
            <style>
            .quick-guide-credit {
                margin-top: 0.85rem;
                color: #64748b;
                font-family: Georgia, "Times New Roman", serif;
                font-size: 0.80rem;
                font-style: italic;
            }
            .quick-guide-credit a { color: #1667b7; }
            .st-key-node-selector-panel { min-height: 76px; }
            </style>
            Explore Boolean-network models through their regulatory structure, Boolean logic, and canalization properties.
            <p>Select a Cell Collective model or upload a <code>.cnet</code> file, adjust the threshold, and click a node to examine its regulators,
            targets, Boolean schemata, canalization map, and node-level parameters.</p>
            <div class="quick-guide-credit">
              Developed at <a href="https://casci.binghamton.edu/casci.php" target="_blank" rel="noopener noreferrer">CASCI Lab</a>
              &nbsp;·&nbsp; Built with <a href="https://github.com/CASCI-lab/CANA" target="_blank" rel="noopener noreferrer">CANA</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_parameter_cards(label: str, icon_svg: str, cards: list[tuple[str, str]]) -> None:
    """Render a compact labelled row of parameter cards."""
    card_html = "".join(
        f'<div class="network-parameter"><span class="parameter-card-copy"><span class="network-parameter-label">{name}</span><span class="network-parameter-value">{value}</span></span></div>'
        for name, value in cards
    )
    st.markdown(
        f'''<div class="model-parameters-row" aria-label="{label}">
          <div class="model-parameters-label"><span class="parameters-label-icon" aria-hidden="true">{icon_svg}</span><span>{label}</span></div>
          <div class="network-parameters">{card_html}</div>
        </div>''',
        unsafe_allow_html=True,
    )

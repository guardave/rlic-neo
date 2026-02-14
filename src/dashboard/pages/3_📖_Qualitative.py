"""
Qualitative Analysis Page.

Displays indicator profiles, economic interpretation, and literature references.
Content is loaded from markdown files in docs/qualitative/.
"""

import re
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar, render_breadcrumb, get_analysis_title

st.set_page_config(page_title="Qualitative | RLIC", page_icon="📖", layout="wide")

# Sidebar: Home, analysis selector, section links
analysis_id = render_sidebar(current_page="Qualitative")

# Content: Breadcrumb, then page
render_breadcrumb("Qualitative")
st.title(f":material/menu_book: Qualitative: {get_analysis_title()}")

# ============================================================================
# Markdown Renderer with Admonition Support
# ============================================================================

CONTENT_DIR = PROJECT_ROOT / "docs" / "qualitative"

# Regex for ::: admonition blocks
ADMONITION_RE = re.compile(
    r'^::: (warning|success|info)\s*\n(.*?)\n^:::\s*$',
    re.MULTILINE | re.DOTALL
)


def render_qualitative(analysis_id: str):
    """Load and render qualitative markdown for an analysis."""
    filepath = CONTENT_DIR / f"{analysis_id}.md"

    if not filepath.exists():
        st.warning(f"Qualitative content for '{analysis_id}' has not been created yet.")
        st.markdown("""
        This analysis is available in the dashboard but detailed qualitative content
        has not yet been added. Please refer to the quantitative tabs (Correlation,
        Lead-Lag, Regimes) for data-driven insights.
        """)
        return

    content = filepath.read_text(encoding='utf-8')

    # Split content into blocks: regular markdown and admonition blocks
    last_end = 0
    for match in ADMONITION_RE.finditer(content):
        # Render any markdown before this admonition
        before = content[last_end:match.start()].strip()
        if before:
            st.markdown(before)

        # Render the admonition
        admonition_type = match.group(1)
        admonition_content = match.group(2).strip()

        if admonition_type == 'warning':
            st.warning(admonition_content)
        elif admonition_type == 'success':
            st.success(admonition_content)
        elif admonition_type == 'info':
            st.info(admonition_content)

        last_end = match.end()

    # Render any remaining markdown after the last admonition
    remaining = content[last_end:].strip()
    if remaining:
        st.markdown(remaining)


# Render content for current analysis
render_qualitative(analysis_id)

# Footer
st.divider()
st.caption("Sources: FRED, Census Bureau, Federal Reserve, NBER, Fidelity, Morningstar, academic literature as cited above.")

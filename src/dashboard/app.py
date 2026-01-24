"""
RLIC Dashboard - Main Entry Point

Redirects to the Home page.

Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="RLIC Dashboard",
    page_icon="📊",
    layout="wide"
)

# Redirect to Home page
st.switch_page("pages/1_🏠_Home.py")

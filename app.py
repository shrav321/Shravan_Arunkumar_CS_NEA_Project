# app.py

import streamlit as st
from db_init import init_db

init_db()

st.set_page_config(page_title="Options Trading Simulator", layout="wide")

st.title("Options Trading Simulator")

st.markdown(
    """
    Use the sidebar to navigate:
    - Market: execute buys and sells
    - Portfolio: view positions and resolve expiry
    - Analysis: advanced modelling
    """
)

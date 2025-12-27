# pages/3_Analysis.py

import streamlit as st
from db_init import init_db

init_db()

st.set_page_config(page_title="Analysis", layout="wide")
st.title("Analysis")

st.info("This page will be implemented in a later cycle.")

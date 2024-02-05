import streamlit as st
from pages.lib import flow

st.title(f"Dataset Graph Construction")

DATA = flow.select_dataset()

st.success(f"Dataset {DATA} is loaded.")
flow.show_graph_statistics()
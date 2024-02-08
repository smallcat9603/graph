import streamlit as st
from pages.lib import flow, cypher

st.title(f"Dataset Graph Construction")

DATA = flow.select_dataset()

st.success(f"Dataset {DATA} is loaded.")
flow.show_graph_statistics()

st.caption("Save graph data including nodes and edges into csv files")
if st.button("Save graph data (.csv)"):
    cypher.save_graph_data(DATA)
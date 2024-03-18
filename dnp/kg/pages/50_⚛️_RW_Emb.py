import streamlit as st
import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds, eigsh
import optuna
from pages.lib import flow


st.title("RW-based Embedding")
st.divider()

graph_tool = st.radio("Select one graph tool:", 
                ["igraph", "networkx"], 
                horizontal=True,
                )
edgefile = st.file_uploader("Choose an edge file:")
labelfile = st.file_uploader("Choose a label file:")
df_label = None
if labelfile is not None:
    df_label = pd.read_csv(labelfile, sep='\s+', header=None) 
if edgefile is not None:
    df = pd.read_csv(edgefile, sep='\s+', header=None)
    ncols = df.shape[1]

    cols = ["source", "target"]
    if ncols > 2:
        cols.append("weight")
    df.columns = cols

    col1_values = df["source"].values
    col2_values = df["target"].values
    min_value = np.min(np.concatenate((col1_values, col2_values)))

    st.divider()
    st.header("Graph Info")
    if graph_tool == "igraph":
        if min_value > 0:
            st.error("Node ID should be from 0!")
            st.stop()
        G = ig.Graph.DataFrame(df, directed=False)
        st.info(G.summary())
    elif graph_tool == "networkx":
        if ncols == 2:
            G = nx.from_pandas_edgelist(df, source="source", target="target")
        elif ncols > 2:
            G = nx.from_pandas_edgelist(df, source="source", target="target", edge_attr="weight")
        st.info(G)

    form = st.form("emb")

    sim = form.radio("Select the similarity metric:", 
                    ["Autocovariance", "PMI"], 
                    horizontal=True,
                    )
    tau = form.slider("Markov time:", 
                    1, 
                    100, 
                    3)
    dim = form.select_slider("Dimension:", 
                             value=128, 
                             options=[128, 256, 512, 1024])
    nrows = form.slider("Number of rows displayed:", 
                    1, 
                    100, 
                    10) 

    if form.form_submit_button("Embedding"):  
        emb_df = flow.node_emb(G, sim=sim, tau=tau, dim=dim, graph_tool=graph_tool, df_label=df_label, nrows=nrows, verbose=True)

        st.header("t-SNE")
        flow.plot_tsne_alt(emb_df)

        st.header("ML")
        ncategories = len(np.unique(emb_df["category"]))
        if ncategories > 1:
            flow.modeler(emb_df)
        else: 
            st.warning(f"The dataset has only one category!")

    if form.form_submit_button("Tune hyperparameters"):
        initial_params = {
            "sim": sim,
            "tau": tau,
            "dim": dim,
            "graph_tool": graph_tool, 
            "df_label": df_label, 
            "nrows": nrows,
        }

        study = optuna.create_study(direction="maximize")
        study.enqueue_trial(initial_params)
        study.optimize(flow.objective_emb(G, graph_tool, df_label, nrows), n_trials=100)

        flow.show_tuning_result(study)
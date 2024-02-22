import streamlit as st
import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds, eigsh
from pages.lib import flow


st.title("RW-based Embedding")
st.divider()

graph_tool = st.radio("Select one graph tool:", 
                ["igraph", "networkx"], 
                horizontal=True,
                # label_visibility="collapsed",
                )
edgefile = st.file_uploader("Choose an edge file:")
if edgefile is not None:
    df = pd.read_csv(edgefile, sep='\s+', header=None)
    st.divider()
    st.header("Graph Info")
    if graph_tool == "igraph":
        if np.min(df.values) > 0:
            st.error("Node ID should be from 0!")
            st.stop()
        G = ig.Graph.DataFrame(df, directed=False)
        st.info(G.summary())
    elif graph_tool == "networkx":
        G = nx.from_pandas_edgelist(df, source=0, target=1)
        st.info(G)

    if st.button("Embedding"):
        nrows = 10
    
        M = flow.standard_random_walk_transition_matrix(G, graph_tool)
        st.header(f"Transition Matrix ({nrows} rows)")
        st.write(M.shape)
        st.table(M[:nrows, :])

        tau = 3 # markov_time
        R = flow.autocovariance_matrix(M, tau)
        st.header(f"Autocovariance Matrix ({nrows} rows)")
        st.write(R.shape)
        st.table(R[:nrows, :])

        R = flow.preprocess_similarity_matrix(R)
        st.header(f"Autocovariance Matrix (clean) ({nrows} rows)")
        st.write(R.shape)
        st.table(R[:nrows, :])

        s, u = eigsh(A=R, k=128, which='LA', maxiter=R.shape[0] * 20)
        st.header(f"Eigenvectors ({nrows} rows)")
        st.write(u.shape)
        st.table(u[:nrows, :])

        u = flow.postprocess_decomposition(u, s)
        st.header(f"Embedding Matrix ({nrows} rows)")
        st.write(u.shape)
        st.table(u[:nrows, :])

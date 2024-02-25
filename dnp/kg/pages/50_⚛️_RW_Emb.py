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
labelfile = st.file_uploader("Choose a label file:")
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

    form = st.form("emb")

    sim = form.radio("Select the similarity metric:", 
                    ["Autocovariance", "PMI"], 
                    horizontal=True,
                    # label_visibility="collapsed",
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
        # adjacency matrix A --> 
        # transition matrix T (= D_1 A) --> 
        # stationary distribution x (via A x = b) --> 
        # autocovariance matrix R (= X M^tau/b -x x^T) --> 
        # eigsh u (via R u = s u) --> 
        # rescale u
    
        M = flow.standard_random_walk_transition_matrix(G, graph_tool=graph_tool)
        st.header(f"Transition Matrix ({nrows} rows)")
        st.write(M.shape)
        st.table(M[:nrows, :])

        if sim == "Autocovariance":
            R = flow.autocovariance_matrix(M, tau)
        elif sim == "PMI":
            R = flow.PMI_matrix(M, tau)
        st.header(f"{sim} Matrix ({nrows} rows)")
        st.write(R.shape)
        st.table(R[:nrows, :])

        R = flow.preprocess_similarity_matrix(R)
        st.header(f"{sim} Matrix (clean) ({nrows} rows)")
        st.write(R.shape)
        st.table(R[:nrows, :])

        s, u = eigsh(A=R, k=dim, which='LA', maxiter=R.shape[0] * 20)
        st.header(f"Eigenvectors ({nrows} rows)")
        st.write(u.shape)
        st.table(u[:nrows, :])

        u = flow.postprocess_decomposition(u, s)
        st.header(f"Embedding Matrix ({nrows} rows)")
        st.write(u.shape)
        st.table(u[:nrows, :])

        st.header("t-SNE")
        n = u.shape[0]
        category = [0]*n
        if labelfile is not None:
            df = pd.read_csv(labelfile, sep='\s+', header=None)
            node_labels = flow.get_node_labels(df)
            category = []
            for i in range(n):
                if 1 in node_labels[i]:  
                    category.append(1)
                else: 
                    category.append(0)
        emb_df = pd.DataFrame(data = {
            "name": range(n),
            "category": category,
            "emb": [row for row in u],
        })
        flow.plot_tsne_alt(emb_df)

        st.header("ML")
        ncategories = len(np.unique(emb_df["category"]))
        if ncategories > 1:
            flow.modeler(emb_df)
        else: 
            st.warning(f"The dataset {st.session_state['data']} has only one category!")
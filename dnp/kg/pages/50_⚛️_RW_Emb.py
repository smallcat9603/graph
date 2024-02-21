import streamlit as st
import networkx as nx
import igraph as ig
import pandas as pd
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
    if graph_tool == "igraph":
        G = ig.Graph.DataFrame(df, directed=False)
    elif graph_tool == "networkx":
        G = nx.from_pandas_edgelist(df, source=0, target=1)
    st.header("Graph Info")
    st.info(G)
   
M = flow.standard_random_walk_transition_matrix(G, graph_tool)
st.header("Transition Matrix")
st.table(M)

tau = 3 # markov_time
R = flow.autocovariance_matrix(M, tau)
st.table(R)

R = flow.preprocess_similarity_matrix(R)
st.table(R)

s, u = eigsh(A=R, k=128, which='LA', maxiter=R.shape[0] * 20)
st.table(u)

u = flow.postprocess_decomposition(u, s)
st.table(u)

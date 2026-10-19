"""Session state and cached wrappers around embkg for the Streamlit pages."""

import os

import streamlit as st
from graphdatascience import GraphDataScience

from embkg import cypher_ops, datasets, gds_ops, paths, rwemb

GRAPH_NAME = "testgraph" # project graph name


def connect():
    """Connect to Neo4j once per session. EMB_NEO4J=SANDBOX selects the sandbox secrets."""
    neo4j_server = "NEO4J_" + os.environ.get("EMB_NEO4J", "LOCAL")
    st.session_state["host"] = st.secrets[neo4j_server]
    st.session_state["user"] = st.secrets[neo4j_server+"_USER"]
    st.session_state["password"] = st.secrets[neo4j_server+"_PASSWORD"]
    st.session_state["gds"] = GraphDataScience(st.session_state["host"], auth=(st.session_state["user"], st.session_state["password"]))
    st.session_state["dir"] = paths.KG_DATA_URL
    st.session_state["graph_name"] = GRAPH_NAME


def gds():
    if "gds" not in st.session_state:
        connect()
    return st.session_state["gds"]


@st.cache_data
def run(query):
    return gds().run_cypher(query)


def require_data():
    if 'data' not in st.session_state:
        st.title("No Graph Data")
        st.warning("You should load graph data first!", icon='⚠')
        st.stop()


def reset():
    gds_ops.drop_memory_graph(gds(), st.session_state["graph_name"])
    cypher_ops.free_up_db(run)

    st.cache_data.clear() # clear cache data via @st.cache_data, not including st.session_state
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def get_emb_result(emb):
    """Read embeddings back uncached: the query text does not change when the embedding
    is recomputed, so the cached `run` would return the first result every time
    (dnp/kg did this; see measurement_log #2)."""
    query = datasets.emb_result_query(st.session_state['data'], emb)
    if query is None:
        st.error("No embedding data is loaded!")
        st.stop()
    return gds().run_cypher(query)


@st.cache_data
def write_nodesimilarity(_G, metric):
    return gds_ops.write_nodesimilarity(gds(), _G, metric)


@st.cache_data
def write_nodesimilarity_ppr(_G, QUERY_DICT):
    return gds_ops.write_nodesimilarity_ppr(gds(), _G, QUERY_DICT)


@st.cache_data
def node_emb_frp(_G, downstream=True, **params):
    gds_ops.node_emb_frp(gds(), _G, downstream=downstream, **params)


@st.cache_data
def node_emb_n2v(_G, downstream=True, **params):
    gds_ops.node_emb_n2v(gds(), _G, downstream=downstream, **params)


@st.cache_data
def knn(_G, emb):
    return gds_ops.knn(gds(), _G, emb)


@st.cache_data
def node_emb(_G, name, sim, tau, dim, graph_tool, df_label, df_label2, alpha, beta, _on_step=None):
    """`name` (the edge file name) keys the cache, since _G is not hashed."""
    return rwemb.node_emb(_G, sim=sim, tau=tau, dim=dim, graph_tool=graph_tool,
                          df_label=df_label, df_label2=df_label2, alpha=alpha, beta=beta, on_step=_on_step)

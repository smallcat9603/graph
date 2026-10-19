import streamlit as st

from embkg import cypher_ops
from ui import core, views

core.require_data()
st.title(f"Node Similarity ({st.session_state['data']})")

if "query" not in st.session_state:
    st.warning(f"No query node in dataset {st.session_state['data']}!")
    st.stop()

##### List in-memory graph

graph_ls = core.gds().graph.list()["graphName"]
if len(graph_ls) > 0:
    for g in graph_ls:
        G = core.gds().graph.get(g)
        st.success(f"Graph {g}: {G.node_count()} nodes and {G.relationship_count()} relationships")
else:
    st.warning('There are currently no graphs in memory.')
    st.stop()

emb_graph = st.selectbox('Enter graph name for node similarity: ', graph_ls)
G = core.gds().graph.get(emb_graph)

if st.session_state["load"] != "Offline":
    results = {metric: core.write_nodesimilarity(G, metric) for metric in ["JACCARD", "OVERLAP", "COSINE"]}
    result_ppr = core.write_nodesimilarity_ppr(G, st.session_state["query"])

    with st.expander("Debug Info"):
        for metric, result in results.items():
            st.header(f"node similarity ({metric})")
            st.write(f"Relationships produced: {result['relationshipsWritten']}")
            st.write(f"Nodes compared: {result['nodesCompared']}")
            st.write(f"Mean similarity: {result['similarityDistribution']['mean']}")

        st.header("node similarity (ppr)")
        st.write(result_ppr)

##############################
### interaction ###
##############################

st.divider()
st.title("UI Interaction")

tabs = {"Node Similarity": lambda: views.interact_node_similarity(st.session_state["query"])}
if st.session_state['data'] == "DNP":
    tabs["Multiple Queries"] = lambda: views.interact_multiple_queries(st.session_state["query"])
tabs["Related Articles"] = views.interact_related_articles
tabs["Common Keywords"] = lambda: views.interact_fixed(cypher_ops.COMMON_KEYWORDS_QUERY)
tabs["Naive by Rank"] = lambda: views.interact_fixed(cypher_ops.NAIVE_BY_RANK_QUERY)
tabs["Naive by Salience"] = lambda: views.interact_fixed(cypher_ops.NAIVE_BY_SALIENCE_QUERY)

for tab, show in zip(st.tabs(list(tabs)), tabs.values()):
    with tab:
        query = show()
        with st.expander("Debug Info"):
            st.code(query)

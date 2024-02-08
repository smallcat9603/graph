import streamlit as st
from pages.lib import cypher, flow

if 'data' not in st.session_state:
   st.title("No Graph Data")
   st.warning("You should load graph data first!", icon='⚠')
   st.stop()
else:
   st.title(f"Node Similarity ({st.session_state['data']})")

if "query" not in st.session_state:
    st.warning(f"No query node in dataset {st.session_state['data']}!")
    st.stop()

##### List in-memory graph

graph_ls = st.session_state["gds"].graph.list()["graphName"]
if len(graph_ls) > 0:
    for g in graph_ls:
        G = st.session_state["gds"].graph.get(g)
        st.success(f"Graph {g}: {G.node_count()} nodes and {G.relationship_count()} relationships")  
else:
    st.warning('There are currently no graphs in memory.')
    st.stop()

##############################
#
#   Node Similarity
#
##############################

emb_graph = st.selectbox('Enter graph name for node similarity: ', graph_ls)
G = st.session_state["gds"].graph.get(emb_graph)

if st.session_state["load"] != "Offline":
    ##############################
    ### node similarity (JACCARD) ###
    ##############################
    result_write_nodesimilarity_jaccard = flow.write_nodesimilarity_jaccard(G)
    ##############################
    ### node similarity (OVERLAP) ###
    ##############################
    result_write_nodesimilarity_overlap = flow.write_nodesimilarity_overlap(G)
    ##############################
    ### node similarity (COSINE) ###
    ##############################
    result_write_nodesimilarity_cosine = flow.write_nodesimilarity_cosine(G)
    ##############################
    ### ppr (personalized pagerank) ###
    ##############################
    result_write_nodesimilarity_ppr = flow.write_nodesimilarity_ppr(G, st.session_state["query"])

    with st.expander("Debug Info"):
        st.header("node similarity (JACCARD)")
        st.write(f"Relationships produced: {result_write_nodesimilarity_jaccard['relationshipsWritten']}")
        st.write(f"Nodes compared: {result_write_nodesimilarity_jaccard['nodesCompared']}")
        st.write(f"Mean similarity: {result_write_nodesimilarity_jaccard['similarityDistribution']['mean']}")

        st.header("node similarity (OVERLAP)")
        st.write(f"Relationships produced: {result_write_nodesimilarity_overlap['relationshipsWritten']}")
        st.write(f"Nodes compared: {result_write_nodesimilarity_overlap['nodesCompared']}")
        st.write(f"Mean similarity: {result_write_nodesimilarity_overlap['similarityDistribution']['mean']}")

        st.header("node similarity (COSINE)")
        st.write(f"Relationships produced: {result_write_nodesimilarity_cosine['relationshipsWritten']}")
        st.write(f"Nodes compared: {result_write_nodesimilarity_cosine['nodesCompared']}")
        st.write(f"Mean similarity: {result_write_nodesimilarity_cosine['similarityDistribution']['mean']}")

        st.header("node similarity (ppr)")
        st.write(result_write_nodesimilarity_ppr)

##############################
### interaction ###
##############################

st.divider()
st.title("UI Interaction")

if st.session_state['data'] == "DNP":
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Node Similarity", "Multiple Queries", "Related Articles", "Common Keywords", "Naive by Rank", "Naive by Salience"])

    with tab1:
        query_interact_node_similarity = cypher.interact_node_similarity(st.session_state["query"])
        with st.expander("Debug Info"):
            st.code(query_interact_node_similarity) 
    with tab2:
        query_interact_multiple_queries = cypher.interact_multiple_queries(st.session_state["query"])
        with st.expander("Debug Info"):
            st.code(query_interact_multiple_queries)
    with tab3:
        query_interact_related_articles = cypher.interact_related_articles()
        with st.expander("Debug Info"):
            st.code(query_interact_related_articles) 
    with tab4:
        query_interact_common_keywords = cypher.interact_common_keywords()
        with st.expander("Debug Info"):
            st.code(query_interact_common_keywords) 
    with tab5:
        query_interact_naive_by_rank = cypher.interact_naive_by_rank()
        with st.expander("Debug Info"):
            st.code(query_interact_naive_by_rank)
    with tab6:
        query_interact_naive_by_salience = cypher.interact_naive_by_salience()
        with st.expander("Debug Info"):
            st.code(query_interact_naive_by_salience)

else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Node Similarity", "Related Articles", "Common Keywords", "Naive by Rank", "Naive by Salience"])

    with tab1:
        query_interact_node_similarity = cypher.interact_node_similarity(st.session_state["query"])
        with st.expander("Debug Info"):
            st.code(query_interact_node_similarity) 
    with tab2:
        query_interact_related_articles = cypher.interact_related_articles()
        with st.expander("Debug Info"):
            st.code(query_interact_related_articles) 
    with tab3:
        query_interact_common_keywords = cypher.interact_common_keywords()
        with st.expander("Debug Info"):
            st.code(query_interact_common_keywords) 
    with tab4:
        query_interact_naive_by_rank = cypher.interact_naive_by_rank()
        with st.expander("Debug Info"):
            st.code(query_interact_naive_by_rank)
    with tab5:
        query_interact_naive_by_salience = cypher.interact_naive_by_salience()
        with st.expander("Debug Info"):
            st.code(query_interact_naive_by_salience)   

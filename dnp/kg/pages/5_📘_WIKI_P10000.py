import streamlit as st
import time
from pages.lib import cypher, param, flow

##############################
### Setting param ###
##############################

DATA = __file__.split("/")[-1].split(".")[0].split("_")[-1]
st.title(f"{DATA} Dataset")
st.info("This database includes wikipedia pages of 10000 persons, consisting of 1000 athletes, 1000 engineers, 1000 actors, 1000 politicians, 1000 physicians, 1000 scientists, 1000 artists, 1000 journalists, 1000 soldiers, and 1000 lawyers.")
nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY = flow.set_param(DATA)

QUERY_DICT = {} # query dict {QUERY_NAME: QUERY_URL}
DATA_URL = f"{param.DATA_DIR}wikidata_persons_10000.csv"  
QUERY_DICT["Joe Biden"] = "https://en.wikipedia.org/wiki/Joe_Biden"

##############################
### Import CSV ###
##############################

cypher.create_constraint()
if DATA_LOAD == "Offline":
    result_import_graph_data = cypher.import_graph_data(DATA)

##############################
### Create Article-[Noun]-Article Graph ###
##############################

st.divider()
st.title("Progress")
progress_bar = st.progress(0, text="Initialize...")
start_time = time.perf_counter()
container_status = st.container(border=False)

##############################
### create url nodes (article, person, ...) ###
##############################

progress_bar.progress(10, text="Create url nodes...")

if DATA_LOAD != "Offline":
    if DATA_TYPE == "URL":
        cypher.load_data_url(DATA_URL)

##############################
### set phrase and salience properties ###
##############################

progress_bar.progress(20, text="Set phrase and salience properties...")

if DATA_LOAD == "Semi-Online":
    result_set_phrase_salience_properties_csv = cypher.set_phrase_salience_properties_csv(f"{param.DATA_DIR}{DATA}.csv")
elif DATA_LOAD == "Online":
    result_set_phrase_salience_properties_gcp = cypher.set_phrase_salience_properties_gcp(GCP_API_KEY)

##############################
### create noun-url relationships ###
##############################

progress_bar.progress(30, text="Create noun-url relationships...")

if DATA_LOAD != "Offline":
    cypher.create_noun_article_relationships(nphrase)

##############################
### query ###
##############################

progress_bar.progress(40, text="Create query nodes...")

if DATA_LOAD != "Offline":
    if DATA_TYPE == "URL":
        for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
            cypher.create_query_node(QUERY_NAME, QUERY_URL)

progress_bar.progress(50, text="Set phrase and salience properties (Query)...")
    
# set phrase and salience properties (Query)
if DATA_LOAD == "Semi-Online":
    cypher.set_phrase_salience_properties_csv(f"{param.DATA_DIR}{DATA}.csv", query_node=True)
elif DATA_LOAD == "Online":
    cypher.set_phrase_salience_properties_gcp(GCP_API_KEY, query_node=True)

progress_bar.progress(60, text="Create noun-article relationships (Query)...")

# create noun-article relationships (Query)
if DATA_LOAD != "Offline":
    cypher.create_noun_article_relationships(nphrase, query_node=True)

##############################
### create article-article relationships ###
##############################

progress_bar.progress(70, text="Create article-article relationships...")

if DATA_LOAD != "Offline":
    cypher.create_article_article_relationships(nphrase)
    cypher.create_article_article_relationships(nphrase, query_node=True)

##############################
### project graph to memory ###
##############################

progress_bar.progress(80, text="Project graph to memory...")

##############################
### graph statistics ###
##############################

G, result = flow.project_graph()

##############################
### node similarity (JACCARD) ###
##############################

if DATA_LOAD != "Offline":
    result_write_nodesimilarity_jaccard = flow.write_nodesimilarity_jaccard(G)

##############################
### node similarity (OVERLAP) ###
##############################

if DATA_LOAD != "Offline":
    result_write_nodesimilarity_overlap = flow.write_nodesimilarity_overlap(G)

##############################
### node similarity (COSINE) ###
##############################

if DATA_LOAD != "Offline":
    result_write_nodesimilarity_cosine = flow.write_nodesimilarity_cosine(G)

##############################
### ppr (personalized pagerank) ###
##############################

if DATA_LOAD != "Offline":
    result_write_nodesimilarity_ppr = flow.write_nodesimilarity_ppr(G, QUERY_DICT)

##############################
### 1. node embedding ###
##############################

progress_bar.progress(90, text="Node embedding...")

if DATA_LOAD != "Offline":
    result_fastRP_stream, result_node2vec_stream, result_hashgnn_stream, result_fastRP_mutate, result_node2vec_mutate, result_hashgnn_mutate = flow.node_embedding(G)

##############################
### 2. kNN ###
##############################

if DATA_LOAD != "Offline":
    result_kNN = flow.kNN(G)

##############################
### evaluate (node embedding + knn) ###
##############################

result_fastrp, result_node2vec, result_hashgnn = cypher.evaluate_embedding_knn()

##############################
### export to csv in import/ ###
##############################

st.divider()

flow.show_graph_statistics()
    
progress_bar.progress(100, text="Finished.")
end_time = time.perf_counter()
execution_time_ms = (end_time - start_time) * 1000
container_status.success(f"Loading finished: {execution_time_ms:.1f} ms. Graph data can be queried.")

st.session_state["data"] = DATA

##############################
### interaction ###
##############################

st.divider()
st.title("UI Interaction")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Node Similarity", "Related Articles", "Common Keywords", "Naive by Rank", "Naive by Salience"])

with tab1:
    query_interact_node_similarity = cypher.interact_node_similarity(QUERY_DICT)
with tab2:
    query_interact_related_articles = cypher.interact_related_articles()
with tab3:
    query_interact_common_keywords = cypher.interact_common_keywords()
with tab4:
    query_interact_naive_by_rank = cypher.interact_naive_by_rank()
with tab5:
    query_interact_naive_by_salience = cypher.interact_naive_by_salience()

st.divider()

##############################
### Verbose ###
##############################

with st.expander("Debug Info"):
    st.header("Data Source")
    st.write(DATA_URL)
    st.header("Query Dict")
    st.table(QUERY_DICT)

    if DATA_LOAD == "Offline":
        st.info("Importing nodes and relationships from csv files finished")
        st.write(result_import_graph_data)

    if DATA_LOAD == "Semi-Online":
        st.header("set phrase salience properties (csv)")
        st.write(result_set_phrase_salience_properties_csv)
    elif DATA_LOAD == "Online":
        st.header("set phrase salience properties (gcp)")
        st.write(result_set_phrase_salience_properties_gcp)

    st.header("Graph Statistics (project)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("# Nodes", str(G.node_count()))
    col2.metric("# Edges", str(G.relationship_count()))
    col3.metric("Density", str(G.density()))
    col4.metric("Memory", str(G.memory_usage()))

    if DATA_LOAD != "Offline":
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

        st.header("1. node embedding")
        st.write(f"Embedding vectors: {result_hashgnn_stream['embedding']}")
        st.write(f"Number of embedding vectors produced: {result_hashgnn_mutate['nodePropertiesWritten']}")

        st.header("2. kNN")
        st.write(f"Relationships produced: {result_kNN['relationshipsWritten']}")
        st.write(f"Nodes compared: {result_kNN['nodesCompared']}")
        st.write(f"Mean similarity: {result_kNN['similarityDistribution']['mean']}")

    st.header("evaluate (fastrp)")
    st.write(result_fastrp)
    st.header("evaluate (node2vec)")
    st.write(result_node2vec)
    st.header("evaluate (hashgnn)")
    st.write(result_hashgnn)

    st.caption("Save graph data including nodes and edges into csv files")
    if st.button("Save graph data (.csv)"):
        cypher.save_graph_data(DATA)

    st.header("query_interact_node_similarity")
    st.code(query_interact_node_similarity) 
    st.header("qquery_interact_related_articles")
    st.code(query_interact_related_articles) 
    st.header("query_interact_common_keywords")
    st.code(query_interact_common_keywords) 
    st.header("query_interact_naive_by_rank")
    st.code(query_interact_naive_by_rank)
    st.header("query_interact_naive_by_salience")
    st.code(query_interact_naive_by_salience)

    st.divider()

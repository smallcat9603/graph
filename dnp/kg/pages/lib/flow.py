import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pages.lib import cypher

def set_param(DATA):
    st.title("Parameters")
    form = st.form("parameters")
    nphrase = form.slider("Number of nouns extracted from each article (50 if Offline is selected)", 
                          1, 
                          100, 
                          50)
    if DATA == "DNP":
        DATA_TYPE = form.radio("Data type", 
                               ["TXT", "URL"], 
                               horizontal=True, 
                               captions=["currently used only for dnp data", "parse html to retrive content"])
    else:
        DATA_TYPE = form.radio("Data type", 
                               ["URL"], 
                               horizontal=True, 
                               captions=["parse html to retrive content"])
    # offline opt: neo4j-admin database dump/load, require to stop neo4j server
    DATA_LOAD = form.radio("Data load", 
                           ["Offline", "Semi-Online", "Online"], 
                           horizontal=True, 
                           captions=["load nodes and relationships from local (avoid to use gcp api, very fast)", "load nodes from local and create relationships during runtime (avoid to use gcp api, fast)", "create nodes and relationships during runtime (use gcp api, slow)"], 
                           index=0)
    GCP_API_KEY = form.text_input('GCP API Key', 
                                  type='password', 
                                  placeholder='should not be empty for Online')
    OUTPUT = form.radio("Output", 
                        ["Simple", "Verbose"], 
                        horizontal=True, 
                        captions=["user mode", "develeper mode (esp. for debug)"])

    run_disabled = False
    if "data" in st.session_state and st.session_state["data"] != DATA:
        run_disabled = True
        form.warning("Please 'Reset' the database status first before you 'Run'!", icon='⚠')
    run = form.form_submit_button("Run", 
                                  type="primary", 
                                  disabled=run_disabled)
    if not run and ("data" not in st.session_state or st.session_state["data"] != DATA):
        st.stop()
    if run and DATA_LOAD == "Online" and GCP_API_KEY == "":
        st.stop()

    return nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY, OUTPUT

def project_graph():
    node_projection = ["Query", "Article", "Noun"]
    # # why raising error "java.lang.UnsupportedOperationException: Loading of values of type StringArray is currently not supported" ???
    # node_projection = {"Query": {"properties": 'phrase'}, "Article": {"properties": 'phrase'}, "Noun": {}}
    relationship_projection = {
        "CONTAINS": {"orientation": "UNDIRECTED", "properties": ["rank", "score", "weight"]},
        # "CORRELATES": {"orientation": "UNDIRECTED", "properties": ["common"]} # Unsupported type [TEXT_ARRAY] of value StringArray[DNP]. Please use a numeric property.
        }
    # # how to project node properties???
    # node_properties = { 
    #     "nodeProperties": {
    #         "phrase": {"defaultValue": []},
    #         "salience": {"defaultValue": []}
    #     }
    # }

    exists_result = st.session_state["gds"].graph.exists(st.session_state["graph_name"])
    if exists_result["exists"]:
        G = st.session_state["gds"].graph.get(st.session_state["graph_name"])
        G.drop()
    G, result = st.session_state["gds"].graph.project(st.session_state["graph_name"], node_projection, relationship_projection)
    # st.title("project graph to memory")
    # st.write(f"The projection took {result['projectMillis']} ms")
    # st.write(f"Graph '{G.name()}' node count: {G.node_count()}")
    # st.write(f"Graph '{G.name()}' node labels: {G.node_labels()}")
    # st.write(f"Graph '{G.name()}' relationship count: {G.relationship_count()}")
    # st.write(f"Graph '{G.name()}' degree distribution: {G.degree_distribution()}")
    # st.write(f"Graph '{G.name()}' density: {G.density()}")
    # st.write(f"Graph '{G.name()}' size in bytes: {G.size_in_bytes()}")
    # st.write(f"Graph '{G.name()}' memory_usage: {G.memory_usage()}")

    return G, result

@st.cache_data
def write_nodesimilarity_jaccard(_G):
    result = st.session_state["gds"].nodeSimilarity.filtered.write(
        _G,
        similarityMetric='JACCARD', # default
        writeRelationshipType='SIMILAR_JACCARD',
        writeProperty='score',
        relationshipWeightProperty="weight",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
        topK=100,
    )
    return result

@st.cache_data
def write_nodesimilarity_overlap(_G):
    result = st.session_state["gds"].nodeSimilarity.filtered.write(
        _G,
        similarityMetric='OVERLAP',
        writeRelationshipType='SIMILAR_OVERLAP',
        writeProperty='score',
        relationshipWeightProperty="weight",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
        topK=100,
    )
    return result

@st.cache_data
def write_nodesimilarity_cosine(_G):
    result = st.session_state["gds"].nodeSimilarity.filtered.write(
        _G,
        similarityMetric='COSINE',
        writeRelationshipType='SIMILAR_COSINE',
        writeProperty='score',
        relationshipWeightProperty="weight",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
        topK=100,
    )
    return result

@st.cache_data
def write_nodesimilarity_ppr(_G, QUERY_DICT):
    for idx, name in enumerate(list(QUERY_DICT.keys())):
        nodeid = st.session_state["gds"].find_node_id(labels=["Query"], properties={"name": name})
        result = st.session_state["gds"].pageRank.write(
            _G,
            writeProperty="pr"+str(idx),
            maxIterations=20,
            dampingFactor=0.85,
            relationshipWeightProperty='weight',
            sourceNodes=[nodeid]
        )
        # if OUTPUT == "Verbose":
        #     st.write(f"Node properties written: {result['nodePropertiesWritten']}")
        #     st.write(f"Mean: {result['centralityDistribution']['mean']}")

@st.cache_data
def node_embedding(_G):
    # fastrp
    result = st.session_state["gds"].fastRP.stream(
        _G,
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight",
        iterationWeights=[1, 1, 1],
    )

    # node2vec
    result = st.session_state["gds"].node2vec.stream(
        _G,
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight",
        iterations=3,
    )

    # hashgnn
    result = st.session_state["gds"].beta.hashgnn.stream(
        _G,
        iterations = 3,
        embeddingDensity = 8,
        generateFeatures = {"dimension": 16, "densityLevel": 1},
        randomSeed = 42,
    )

    # if OUTPUT == "Verbose":
    #     st.write(f"Embedding vectors: {result['embedding']}")

    # fastrp
    result = st.session_state["gds"].fastRP.mutate(
        _G,
        mutateProperty="embedding_fastrp",
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight", # each relationship should have
        iterationWeights=[1, 1, 1],
    )

    # node2vec
    result = st.session_state["gds"].node2vec.mutate(
        _G,
        mutateProperty="embedding_node2vec",
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight",
        iterations=3,
    )

    # hashgnn
    result = st.session_state["gds"].beta.hashgnn.mutate(
        _G,
        mutateProperty="embedding_hashgnn",
        randomSeed=42,
        heterogeneous=True,
        iterations=3,
        embeddingDensity=8,
        # opt1
        generateFeatures={"dimension": 16, "densityLevel": 1},
        # # opt2 not work
        # binarizeFeatures={"dimension": 16, "threshold": 0},
        # featureProperties=['phrase', 'salience'], # each node should have
    )

    return result

@st.cache_data
def kNN(_G):
    # fastrp
    result = st.session_state["gds"].knn.filtered.write(
        _G,
        topK=10,
        nodeProperties=["embedding_fastrp"],
        randomSeed=42, # Note that concurrency must be set to 1 when setting this parameter.
        concurrency=1,
        sampleRate=1.0,
        deltaThreshold=0.0,
        writeRelationshipType="SIMILAR_FASTRP",
        writeProperty="score",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
    )

    # node2vec
    result = st.session_state["gds"].knn.filtered.write(
        _G,
        topK=10,
        nodeProperties=["embedding_node2vec"],
        randomSeed=42, # Note that concurrency must be set to 1 when setting this parameter.
        concurrency=1,
        sampleRate=1.0,
        deltaThreshold=0.0,
        writeRelationshipType="SIMILAR_NODE2VEC",
        writeProperty="score",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
    )

    # hashgnn
    result = st.session_state["gds"].knn.filtered.write(
        _G,
        topK=10,
        nodeProperties=["embedding_hashgnn"],
        randomSeed=42, # Note that concurrency must be set to 1 when setting this parameter.
        concurrency=1,
        sampleRate=1.0,
        deltaThreshold=0.0,
        writeRelationshipType="SIMILAR_HASHGNN",
        writeProperty="score",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
    )

    return result

def show_graph_statistics():
    st.title("Graph Statistics")
    result = cypher.get_graph_statistics()

    col1, col2 = st.columns(2)
    col1.metric("# Nodes", result["nodeCount"][0])
    col2.metric("# Edges", result["relCount"][0])

    col1, col2 = st.columns(2)
    with col1.expander("Node Labels"):
        st.table(result["labels"][0])
    with col2.expander("Relationship Types"):
        st.table(result["relTypesCount"][0])

@st.cache_data
def plot_similarity(result, query_node, similarity_method, limit):
    fig, ax = plt.subplots()
    articles = result["Article"]
    y_pos = np.arange(len(articles))
    similarities = result["Similarity"]

    ax.barh(y_pos, similarities)
    ax.set_yticks(y_pos, labels=articles)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Similarity Score')
    ax.set_title(f'Similarity to {query_node} by {similarity_method} (Top-{limit})')

    st.pyplot(fig)

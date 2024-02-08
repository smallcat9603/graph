import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import time
import re
from collections import Counter
from sklearn.manifold import TSNE
import altair as alt
from pages.lib import cypher

def select_data():

    TYPE = st.radio("Select one data type:", 
                    ["DNP", "WIKI", "CYPHER"], 
                    horizontal=True,
                    # label_visibility="collapsed",
                    )

    if TYPE == "DNP":
        DATA = st.radio("Select one dataset", 
                        ["DNP"], 
                        captions=["This database includes 100 DNP newsreleases, and 4 Toppan newsreleases."],
                        # label_visibility="collapsed",
                        )
        LANGUAGE = "ja"

    elif TYPE == "WIKI":
        DATA = st.radio("Select one dataset", 
                        ["FP100", "P100", "P1000", "P10000"], 
                        captions=["This database includes wikipedia pages of 100 football players.",
                                  "This database includes wikipedia pages of 100 persons, consisting of 25 athletes, 25 engineers, 25 actors, and 25 politicians.",
                                  "This database includes wikipedia pages of 1000 persons, consisting of 100 athletes, 100 engineers, 100 actors, 100 politicians, 100 physicians, 100 scientists, 100 artists, 100 journalists, 100 soldiers, and 100 lawyers.",
                                  "This database includes wikipedia pages of 10000 persons, consisting of 1000 athletes, 1000 engineers, 1000 actors, 1000 politicians, 1000 physicians, 1000 scientists, 1000 artists, 1000 journalists, 1000 soldiers, and 1000 lawyers."]
                        )
        LANGUAGE = "en"

    elif TYPE == "CYPHER":
        DATA = st.radio("Select one dataset", 
                        ["euro_roads"], 
                        captions=["The dataset contains 894 towns, 39 countries, and 1,250 roads connecting them."]
                        )
        LANGUAGE = "en"

    return TYPE, DATA, LANGUAGE

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
                           ["Offline", "Semi-Online", "Online", "On-the-fly"], 
                           horizontal=True, 
                           captions=["load nodes and relationships from local (avoid to use gcp api, very fast)", "load nodes from local and create relationships during runtime (avoid to use gcp api, fast)", "create nodes and relationships during runtime (use gcp api, slow)", "use spaCy to extract keywords from each article (free)"], 
                           index=0)
    col1, col2 = form.columns(2)
    expander_gcp_api_key = col1.expander("GCP API Key (Mandatory for Online)")
    GCP_API_KEY = expander_gcp_api_key.text_input("GCP API Key", 
                                type="password", 
                                placeholder="should not be empty for Online",
                                label_visibility="collapsed")
    expander_keywords = col2.expander("Keywords (Optional for On-the-fly)")
    WORD_CLASS = expander_keywords.multiselect("Keywords",
                                             ["NOUN", "ADJ", "VERB"], 
                                             ["NOUN"])
    PIPELINE_SIZE = expander_keywords.radio("Pipeline size", 
                           ["Small", "Medium", "Large"], 
                           horizontal=True, 
                           captions=["10MB+", "40MB+", "500MB+"])

    run_disabled = False
    if "data" in st.session_state and st.session_state["data"] != DATA:
        run_disabled = True
        form.warning("Please 'Reset' the database status first before you 'Run'!", icon='⚠')

    if form.form_submit_button("Run", type="primary", disabled=run_disabled):
        if DATA_LOAD == "Online" and GCP_API_KEY == "":
            form.warning("Please input GCP API Key (Mandatory for Online) before you 'Run'!", icon='⚠')
            st.stop()
        elif DATA_LOAD == "On-the-fly" and len(WORD_CLASS) == 0:
            form.warning("Please select at least one keyword type before you 'Run'!", icon='⚠')
            st.stop()
    else:
        if "data" not in st.session_state or st.session_state["data"] != DATA:
            st.stop()

    return nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY, WORD_CLASS, PIPELINE_SIZE

def project_graph(node_label_list, relationship_type, relationship_property_list):
    node_projection = node_label_list
    # # why raising error "java.lang.UnsupportedOperationException: Loading of values of type StringArray is currently not supported" ???
    # # update: only numerical property values are supported
    # node_projection = {"Query": {"properties": 'phrase'}, "Article": {"properties": 'phrase'}, "Noun": {}}
    relationship_projection = {
        relationship_type: {"orientation": "UNDIRECTED", "properties": relationship_property_list},
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
    results = ""
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
        results += f"Node properties written: {result['nodePropertiesWritten']}\n"
        results += f"Mean: {result['centralityDistribution']['mean']}\n"
    return results

@st.cache_data
def node_emb_frp(_G, embeddingDimension, normalizationStrength, iterationWeights, nodeSelfInfluence, relationshipWeightProperty, randomSeed, propertyRatio, featureProperties):
    st.session_state["gds"].fastRP.mutate( # for downstream knn
        _G,
        embeddingDimension=embeddingDimension,
        normalizationStrength=normalizationStrength,
        iterationWeights=iterationWeights,
        nodeSelfInfluence=nodeSelfInfluence,
        relationshipWeightProperty=relationshipWeightProperty,
        randomSeed=randomSeed,
        propertyRatio=propertyRatio,
        featureProperties=featureProperties,
        mutateProperty="emb_frp"            
    )
    st.session_state["gds"].fastRP.write( # for t-SNE
        _G,
        embeddingDimension=embeddingDimension,
        normalizationStrength=normalizationStrength,
        iterationWeights=iterationWeights,
        nodeSelfInfluence=nodeSelfInfluence,
        relationshipWeightProperty=relationshipWeightProperty,
        randomSeed=randomSeed,
        propertyRatio=propertyRatio,
        featureProperties=featureProperties,
        writeProperty="emb_frp"          
    )

@st.cache_data
def node_emb_n2v(_G, embeddingDimension, walkLength, walksPerNode, inOutFactor, returnFactor, negativeSamplingRate, iterations, initialLearningRate, minLearningRate, walkBufferSize, relationshipWeightProperty,randomSeed):
    st.session_state["gds"].node2vec.mutate( # for downstream knn
        _G,
        embeddingDimension=embeddingDimension,
        walkLength=walkLength,
        walksPerNode=walksPerNode,
        inOutFactor=inOutFactor,
        returnFactor=returnFactor,
        negativeSamplingRate=negativeSamplingRate,
        iterations=iterations,
        initialLearningRate=initialLearningRate,
        minLearningRate=minLearningRate,
        walkBufferSize=walkBufferSize,
        relationshipWeightProperty=relationshipWeightProperty,
        randomSeed=randomSeed,
        mutateProperty="emb_n2v",           
    )
    st.session_state["gds"].node2vec.write( # for t-SNE
        _G,
        embeddingDimension=embeddingDimension,
        walkLength=walkLength,
        walksPerNode=walksPerNode,
        inOutFactor=inOutFactor,
        returnFactor=returnFactor,
        negativeSamplingRate=negativeSamplingRate,
        iterations=iterations,
        initialLearningRate=initialLearningRate,
        minLearningRate=minLearningRate,
        walkBufferSize=walkBufferSize,
        relationshipWeightProperty=relationshipWeightProperty,
        randomSeed=randomSeed,
        writeProperty="emb_n2v",           
    )
                    
# not in use
@st.cache_data
def node_emb(_G):
    # fastrp
    result_fastRP_stream = st.session_state["gds"].fastRP.stream(
        _G,
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight",
        iterationWeights=[1, 1, 1],
    )

    # node2vec
    result_node2vec_stream = st.session_state["gds"].node2vec.stream(
        _G,
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight",
        iterations=3,
    )

    # hashgnn
    result_hashgnn_stream = st.session_state["gds"].beta.hashgnn.stream(
        _G,
        iterations = 3,
        embeddingDensity = 8,
        generateFeatures = {"dimension": 16, "densityLevel": 1},
        randomSeed = 42,
    )

    # fastrp
    result_fastRP_mutate = st.session_state["gds"].fastRP.mutate(
        _G,
        mutateProperty="embedding_fastrp",
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight", # each relationship should have
        iterationWeights=[1, 1, 1],
    )

    # node2vec
    result_node2vec_mutate = st.session_state["gds"].node2vec.mutate(
        _G,
        mutateProperty="embedding_node2vec",
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight",
        iterations=3,
    )

    # hashgnn
    result_hashgnn_mutate = st.session_state["gds"].beta.hashgnn.mutate(
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

    return result_fastRP_stream, result_node2vec_stream, result_hashgnn_stream, result_fastRP_mutate, result_node2vec_mutate, result_hashgnn_mutate

@st.cache_data
def kNN(_G, emb, topK=100, writeProperty="score", sourceNodeFilter="Query", targetNodeFilter="Article"):
    if emb == "emb_frp": # fastrp
        writeRelationshipType = "SIMILAR_FASTRP"
    elif emb == "emb_n2v": # node2vec  
        writeRelationshipType = "SIMILAR_NODE2VEC" 
    elif emb == "emb_hgn":
        writeRelationshipType="SIMILAR_HASHGNN"

    result = st.session_state["gds"].knn.filtered.write(
        _G,
        topK=topK,
        nodeProperties=[emb], # in-memory (used mutate, not write)
        randomSeed=42, # Note that concurrency must be set to 1 when setting this parameter.
        concurrency=1,
        sampleRate=1.0,
        deltaThreshold=0.0,
        writeRelationshipType=writeRelationshipType,
        writeProperty=writeProperty,
        sourceNodeFilter=sourceNodeFilter,
        targetNodeFilter=targetNodeFilter,
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

@st.cache_data
def get_nlp(language, pipeline_size):
    if language == "en":
        if pipeline_size == "Small":
            nlp = spacy.load("en_core_web_sm")
        elif pipeline_size == "Medium":
            nlp = spacy.load("en_core_web_md")
        elif pipeline_size == "Large":
            nlp = spacy.load("en_core_web_lg")
    elif language == "ja":
        if pipeline_size == "Small":
            nlp = spacy.load("ja_core_news_sm")
        elif pipeline_size == "Medium":
            nlp = spacy.load("ja_core_news_md")
        elif pipeline_size == "Large":
            nlp = spacy.load("ja_core_news_lg") 

    return nlp

@st.cache_data
def extract_keywords(_nlp, text, word_class, n):
    doc = _nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in word_class]
    keyword_freq = Counter(keywords)
    top_keywords = keyword_freq.most_common(n) 
    
    return top_keywords

@st.cache_data
def get_nodes_relationships_csv(file):
    df = pd.read_csv(file)
    header_node = "_labels"
    header_relationship = "_type"

    nodes = df[header_node].unique().tolist()
    nodes = [value for value in nodes if isinstance(value, str)]
    nodes = [value[1:] if value.startswith(":") else value for value in nodes]

    relationships = df[header_relationship].unique().tolist()
    relationships = [value for value in relationships if isinstance(value, str)]
    relationships = [value[1:] if value.startswith(":") else value for value in relationships]

    return nodes, relationships

def drop_memory_graph(graph_name):
    exists_result = st.session_state["gds"].graph.exists(graph_name)
    if exists_result["exists"]:
        G = st.session_state["gds"].graph.get(graph_name)
        G.drop()  

def plot_tsne_alt(result):
    X = np.array(list(result["emb"]))
    X_embedded = TSNE(n_components=2, random_state=6).fit_transform(X)

    names = result["name"]
    categories = result["category"]
    tsne_df = pd.DataFrame(data = {
        "name": names,
        "category": categories,
        "x": [value[0] for value in X_embedded],
        "y": [value[1] for value in X_embedded],
    })

    chart = alt.Chart(tsne_df).mark_circle(size=60).encode(
    x="x",
    y="y",
    color="category",
    tooltip=["name", "category"],
    ).properties(width=700, height=400)

    st.altair_chart(chart, use_container_width=True)

def update_state(DATA, DATA_LOAD, QUERY_DICT):
    st.session_state["data"] = DATA
    st.session_state["load"] = DATA_LOAD
    st.session_state["query"] = QUERY_DICT

def construct_graph_cypher(DATA, LANGUAGE, DATA_URL, QUERY_DICT, nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY, WORD_CLASS, PIPELINE_SIZE):
    ##############################
    ### Import CSV ###
    ##############################

    cypher.create_constraint(st.session_state["constraint"])
    if DATA_LOAD == "Offline":
        result_import_graph_data = cypher.import_graph_data(DATA)

    ##############################
    ### Create Article-[Noun]-Article Graph ###
    ##############################

    st.divider()
    st.title(f"Progress ({DATA})")
    progress_bar = st.progress(0, text="Initialize...")
    start_time = time.perf_counter()
    container_status = st.container(border=False)

    ##############################
    ### create url nodes (article, person, ...) ###
    ##############################

    progress_bar.progress(20, text="Create url nodes...")

    if DATA_LOAD != "Offline":
        if DATA_TYPE == "TXT":
            for idx in range(1, 101):
                node = "B-" + str(idx)
                file = DATA_URL + node + ".txt"
                content = ""
                with open(file, 'r') as f:
                    content = f.read()
                    content = re.sub('\n+', ' ', content)
                query = f"""
                MERGE (a:Article {{ name: "{node}", url: "{file}", body: "{content}" }})
                """
                cypher.run(query)
            # query
            for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
                content = ""
                with open(QUERY_URL, 'r') as f:
                    content = f.read()
                    content = re.sub('\n+', ' ', content)
                query = f"""
                MERGE (q:Query {{ name: "{QUERY_NAME}", url: "{QUERY_URL}", body: "{content}" }})
                """
                cypher.run(query)
        else:
            cypher.load_data_url(DATA_URL)
            # query
            for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
                cypher.create_query_node(QUERY_NAME, QUERY_URL)

    ##############################
    ### set phrase and salience properties ###
    ##############################
    
    progress_bar.progress(40, text="Set phrase and salience properties...")

    if DATA_LOAD == "Semi-Online":
        result_set_phrase_salience_properties_csv = cypher.set_phrase_salience_properties_csv(f"{st.session_state['dir']}{DATA}.csv")
        cypher.set_phrase_salience_properties_csv(f"{st.session_state['dir']}{DATA}.csv", query_node=True)
    elif DATA_LOAD == "Online":
        result_set_phrase_salience_properties_gcp = cypher.set_phrase_salience_properties_gcp(GCP_API_KEY)
        cypher.set_phrase_salience_properties_gcp(GCP_API_KEY, query_node=True)
    elif DATA_LOAD == "On-the-fly":
        result_set_phrase_salience_properties_spacy = cypher.set_phrase_salience_properties_spacy(LANGUAGE, WORD_CLASS, PIPELINE_SIZE, nphrase, query_node=False)
        cypher.set_phrase_salience_properties_spacy(LANGUAGE, WORD_CLASS, PIPELINE_SIZE, nphrase, query_node=True)

    ##############################
    ### create noun-url relationships ###
    ##############################

    progress_bar.progress(60, text="Create noun-url relationships...")

    if DATA_LOAD != "Offline":
        cypher.create_noun_article_relationships(nphrase)
        cypher.create_noun_article_relationships(nphrase, query_node=True)
    
    ##############################
    ### create article-article relationships ###
    ##############################

    progress_bar.progress(80, text="Create article-article relationships...")

    if DATA_LOAD != "Offline":
        cypher.create_article_article_relationships(nphrase)
        cypher.create_article_article_relationships(nphrase, query_node=True)

    ##############################
    ### state update ###
    ##############################
        
    update_state(DATA, DATA_LOAD, QUERY_DICT)

    ##############################
    ### export to csv in import/ ###
    ##############################

    progress_bar.progress(100, text="Finished. Show graph statistics...")

    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) * 1000
    container_status.success(f"Loading finished: {execution_time_ms:.1f} ms. Graph data can be queried.")

    st.divider()
    show_graph_statistics()

    st.caption("Save graph data including nodes and edges into csv files")
    if st.button("Save graph data (.csv)"):
        cypher.save_graph_data(DATA)

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
        elif DATA_LOAD == "On-the-fly":
            st.header("set phrase salience properties (spacy)")
            st.write(result_set_phrase_salience_properties_spacy)

def construct_graph_cypherfile(DATA, LANGUAGE):
    
    run_disabled = False
    if "data" in st.session_state and st.session_state["data"] != DATA:
        run_disabled = True
        st.warning("Please 'Reset' the database status first before you 'Run'!", icon='⚠')

    if st.button("Run", type="primary", disabled=run_disabled):
        if DATA == "euro_roads":
            file_cypher = "https://raw.githubusercontent.com/smallcat9603/graph/main/cypher/euro_roads.cypher"

        cypher.runFile(file_cypher)
        st.session_state["data"] = DATA
    else:
        if "data" not in st.session_state or st.session_state["data"] != DATA:
            st.stop()

    st.success(f"Dataset {DATA} is loaded.")
    show_graph_statistics()

    st.caption("Save graph data including nodes and edges into csv files")
    if st.button("Save graph data (.csv)"):
        cypher.save_graph_data(DATA)

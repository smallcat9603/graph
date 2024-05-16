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
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import optuna
import networkx as nx
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds, eigsh
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
        DATA = st.radio("Select one dataset:", 
                        ["euro_roads", "newfood", "blogcatalog", "airport", "airport0.8"], 
                        captions=["The dataset contains 894 towns, 39 countries, and 1,250 roads connecting them. https://github.com/neo4j-examples/graph-embeddings/raw/main/data/roads.csv",
                                  "The dataset contains nutritional information alongside the ingredients used in 1600+ dishes. https://raw.githubusercontent.com/smallcat9603/graph/main/data/newfood.csv",
                                  "The dataset contains undirected social network of bloggers with (multi) labels representing topics of interest. https://raw.githubusercontent.com/smallcat9603/graph/main/data/blogcatalog_0.edges",
                                  "The dataset contains flight network among global airports (from Open-Flights). https://raw.githubusercontent.com/smallcat9603/graph/main/data/airport_0.edges",
                                  "The dataset contains flight network among global airports (from Open-Flights), however 20 percent of edges are randomly removed while keeping graph connected. https://raw.githubusercontent.com/smallcat9603/graph/main/data/airport_0_remaining.edges"]
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
def node_emb_frp(_G, embeddingDimension, normalizationStrength, iterationWeights, nodeSelfInfluence, relationshipWeightProperty, randomSeed, propertyRatio, featureProperties, downstream=True):
    if downstream:
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
def node_emb_n2v(_G, embeddingDimension, walkLength, walksPerNode, inOutFactor, returnFactor, negativeSamplingRate, iterations, initialLearningRate, minLearningRate, walkBufferSize, relationshipWeightProperty,randomSeed, downstream=True):
    if downstream:
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

@st.cache_data
def node_emb(_G, sim, tau, dim, graph_tool, df_label, df_label2, alpha, beta, verbose=False, save_emb=False, name="test"):
    # adjacency matrix A --> 
    # transition matrix T (= D_1 A) --> 
    # stationary distribution x (via A x = b) --> 
    # autocovariance matrix R (= X M^tau/b -x x^T) --> 
    # eigsh u (via R u = s u) --> 
    # rescale u        

    nrows = 10

    M = standard_random_walk_transition_matrix(_G, graph_tool=graph_tool)
    if verbose:
        st.header(f"Transition Matrix ({nrows} rows)")
        st.write(M.shape)
        st.table(M[:nrows, :])

    n = M.shape[0]
    category = [0]*n
    if df_label is not None:
        node_labels = get_node_labels(df_label)
        category = get_category_list(node_labels)

        node_labels2 = None
        if df_label2 is not None:
            node_labels2 = get_node_labels(df_label2)
        M = M_attr(n=n, M=M, node_labels=node_labels, node_labels2=node_labels2, alpha=alpha, beta=beta)

        # st.write(np.sum(M, axis=1))
        st.header(f"Transition Matrix with Attributes ({nrows} rows)")
        st.write(M.shape)
        st.table(M[:nrows, :])

    if sim == "Autocovariance":
        R = autocovariance_matrix(M, tau)
    elif sim == "PMI":
        R = PMI_matrix(M, tau)
    if verbose:
        st.header(f"{sim} Matrix ({nrows} rows)")
        st.write(R.shape)
        st.table(R[:nrows, :])

    R = preprocess_similarity_matrix(R)
    if verbose:
        st.header(f"{sim} Matrix (clean) ({nrows} rows)")
        st.write(R.shape)
        st.table(R[:nrows, :])

    s, u = eigsh(A=R, k=dim, which='LA', maxiter=R.shape[0] * 20)
    if verbose:
        st.header(f"Eigenvectors ({nrows} rows)")
        st.write(u.shape)
        st.table(u[:nrows, :])

    u = postprocess_decomposition(u, s)
    if verbose:
        st.header(f"Embedding Matrix ({nrows} rows)")
        st.write(u.shape)
        st.table(u[:nrows, :])

    emb_df = pd.DataFrame(data = {
        "name": range(n),
        "category": category,
        "emb": [row for row in u],
    }) 

    if save_emb:
        np.savetxt(f"/Users/smallcat/Documents/GitHub/graph/dnp/kg/emb/{name}_tau{tau}_dim{dim}_alpha{alpha}_beta{beta}.txt", u)

    return emb_df 

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

    cypher.create_constraint("ID_UNIQUE")
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
        elif DATA == "newfood":
            file_cypher = "https://raw.githubusercontent.com/smallcat9603/graph/main/cypher/newfood.cypher"
        elif DATA == "blogcatalog":
            file_cypher = "https://raw.githubusercontent.com/smallcat9603/graph/main/cypher/blogcatalog.cypher"
        elif DATA == "airport":
            file_cypher = "https://raw.githubusercontent.com/smallcat9603/graph/main/cypher/airport.cypher"
        elif DATA == "airport0.8":
            file_cypher = "https://raw.githubusercontent.com/smallcat9603/graph/main/cypher/airport_removed.cypher"

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

def create_X_y(result):

    names = result["name"]
    categories = result["category"]
    embs = result["emb"]

    emb_df = pd.DataFrame(data = {
        "name": names,
        "category": categories,
        "emb": embs,
    })

    emb_df['target'] = pd.factorize(emb_df['category'])[0].astype("float32")
    y = emb_df['target'].to_numpy()
    emb_df['X'] = emb_df['emb'].apply(lambda x: np.array(x))
    X = np.array(emb_df['X'].to_list())

    return X, y

def modeler(result, k_folds=5, model='linear', show_matrix=True):

    acc_scores = []

    X, y = create_X_y(result)

    for i in range(k_folds):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = svm.SVC(kernel=model, class_weight='balanced')
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc = accuracy_score(pred, y_test)
        acc_scores.append(acc)        
    
    if show_matrix:
        st.write('Accuracy scores: ', acc_scores)
        st.write('Mean accuracy: ', np.mean(acc_scores))

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, pred, labels=clf.classes_, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(ax=ax)
        st.pyplot(fig)
    else:
        return np.mean(acc_scores)

def objective_frp(G, rel_weight_prop, prop_rat, feat_prop):
    def objective(trial):
        params = {
            "embeddingDimension": trial.suggest_int("embeddingDimension", 128, 512, log=True),
            "normalizationStrength": trial.suggest_float("normalizationStrength", -1.0, 1.0),
            "iterationWeights1": trial.suggest_float("iterationWeights1", 0.0, 1.0),
            "iterationWeights2": trial.suggest_float("iterationWeights2", 0.0, 1.0),
            "iterationWeights3": trial.suggest_float("iterationWeights3", 0.0, 1.0),
            "nodeSelfInfluence": trial.suggest_float("nodeSelfInfluence", 0.0, 1.0),
            "randomSeed": trial.suggest_int("randomSeed", 1, 99),
        }

        frp_dim = params["embeddingDimension"]
        frp_norm = params["normalizationStrength"]
        frp_it_weight1 = params["iterationWeights1"]
        frp_it_weight2 = params["iterationWeights2"]
        frp_it_weight3 = params["iterationWeights3"]
        node_self_infl = params["nodeSelfInfluence"]
        frp_seed = params["randomSeed"]

        node_emb_frp(G, 
            embeddingDimension=frp_dim,
            normalizationStrength=frp_norm,
            iterationWeights=[frp_it_weight1,frp_it_weight2,frp_it_weight3],
            nodeSelfInfluence=node_self_infl,
            relationshipWeightProperty=rel_weight_prop,
            randomSeed=frp_seed,
            propertyRatio=prop_rat,
            featureProperties=feat_prop,
            downstream=False
        )

        result = cypher.get_emb_result("emb_frp")

        return modeler(result, show_matrix=False)
    
    return objective

def objective_n2v(G, rel_weight_prop):
    def objective(trial):
        params = {
            "embeddingDimension": trial.suggest_int("embeddingDimension", 128, 512, log=True),
            "walkLength": trial.suggest_int("walkLength", 2, 160),
            "walksPerNode": trial.suggest_int("walksPerNode", 2, 50),
            "inOutFactor": trial.suggest_float("inOutFactor", 0.001, 1.0, step=0.05),
            "returnFactor": trial.suggest_float("returnFactor", 0.001, 1.0, step=0.05),
            "negativeSamplingRate": trial.suggest_int("negativeSamplingRate", 5, 20),
            "iterations": trial.suggest_int("iterations", 1, 10),
            # "initialLearningRate": trial.suggest_float("initialLearningRate", 0.001, 0.1),
            "initialLearningRate": trial.suggest_categorical("initialLearningRate", [0.001, 0.005, 0.01, 0.05, 0.1]),
            # "minLearningRate": trial.suggest_float("minLearningRate", 0.0001, 0.01),
            "minLearningRate": trial.suggest_categorical("minLearningRate", [0.0001, 0.0005, 0.001, 0.005, 0.01]),
            "walkBufferSize": trial.suggest_int("walkBufferSize", 100, 2000),
            "randomSeed": trial.suggest_int("randomSeed", 1, 99),
        }

        n2v_dim = params["embeddingDimension"]
        n2v_walk_length = params["walkLength"]
        n2v_walks_node = params["walksPerNode"]
        n2v_io_factor = params["inOutFactor"]
        n2v_ret_factor = params["returnFactor"]
        n2v_neg_samp_rate = params["negativeSamplingRate"]
        n2v_iterations = params["iterations"]
        n2v_init_lr = params["initialLearningRate"]
        n2v_min_lr = params["minLearningRate"]
        n2v_walk_bs = params["walkBufferSize"]
        n2v_seed  = params["randomSeed"]
        
        node_emb_n2v(
            G,
            embeddingDimension=n2v_dim,
            walkLength=n2v_walk_length,
            walksPerNode=n2v_walks_node,
            inOutFactor=n2v_io_factor,
            returnFactor=n2v_ret_factor,
            negativeSamplingRate=n2v_neg_samp_rate,
            iterations=n2v_iterations,
            initialLearningRate=n2v_init_lr,
            minLearningRate=n2v_min_lr,
            walkBufferSize=n2v_walk_bs,
            relationshipWeightProperty=rel_weight_prop,
            randomSeed=n2v_seed,
            downstream=False         
        )

        result = cypher.get_emb_result("emb_n2v")

        return modeler(result, show_matrix=False)
    
    return objective

def objective_emb(G, graph_tool, df_label, df_label2):
    def objective(trial):
        params = {
            "sim": trial.suggest_categorical("sim", ["Autocovariance", "PMI"]),
            "tau": trial.suggest_int("tau", 1, 100),
            "dim": trial.suggest_int("dim", 128, 1024, log=True),
            "alpha": trial.suggest_float("alpha", 0.1, 1.0, step=0.1),
            "beta": trial.suggest_float("beta", 0.1, 1.0, step=0.1),
        }

        sim = params["sim"]
        tau = params["tau"]
        dim = params["dim"]
        alpha = params["alpha"]
        beta = params["beta"]

        result = node_emb(G,
                        sim=sim,
                        tau=tau,
                        dim=dim,
                        graph_tool=graph_tool,
                        df_label=df_label,  
                        df_label2=df_label2,
                        alpha=alpha,
                        beta=beta,     
                        )

        return modeler(result, show_matrix=False)
    
    return objective

def show_tuning_result(study):
    st.write(f"Accuracy: {study.best_trial.value}")
    st.write(f"Best hyperparameters: {study.best_trial.params}")

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()


def standard_random_walk_transition_matrix(G, graph_tool="igraph"):
    """
    Transition matrix for the standard random-walk given the input graph.

    :param G: Input graph.
    :return: Standard random-walk transition matrix.
    """

    """
    e.g, 0-1, 0-2, 1-2, 2-3
    degree_vector = np.array(G.degree()) = [2 2 3 1]
    1/degree_vector = [0.5 0.5 0.33 1.]
    """

    if graph_tool == "igraph":
        degree_vector = np.array(G.degree())
        D_1 = np.diag(1/degree_vector)
        A = np.array(G.get_adjacency().data)
    elif graph_tool == "networkx":
        degree_vector = np.array([a[1] for a in sorted(G.degree(weight='weight'), key=lambda a: a[0])])
        D_1 = np.diag(1/degree_vector)
        A = nx.adjacency_matrix(G, sorted(G.nodes)).toarray()
        
    return np.matmul(D_1, A)

def stationary_distribution(M):
    """
    Stationary distribution given the transition matrix.

    :param M: Transition matrix.
    :return: Stationary distribution.
    """

    # We solve (M^T - I) x = 0 and 1 x = 1. Combine them and let A = [M^T - I; 1], b = [0; 1]. We have A x = b.
    n = M.shape[0]
    A = np.concatenate([M.T - np.identity(n), np.ones(shape=(1,n))], axis=0)
    b = np.concatenate([np.zeros(n), [1]], axis=0)

    # Solve A^T A x = A^T b instead (since A is not square).
    x = np.linalg.solve(A.T @ A, A.T @ b)

    return x

def autocovariance_matrix(M, tau, b=1):
    """
    Autocovariance matrix given a transition matrix. X M^tau/b -x x^T

    :param M: Transition matrix.
    :param tau: Markov time.
    :param b: Number of negative samples used in the sampling algorithm.
    :return: Autocovariance matrix.
    """

    x = stationary_distribution(M)
    X = np.diag(x)
    M_tau = np.linalg.matrix_power(M, tau)

    return X @ M_tau/b - np.outer(x, x) 

def PMI_matrix(M, tau, b=1):
    """
    PMI matrix given a transition matrix. log(X M^tau/b) - log(x x^T)

    :param M: transition matrix
    :param tau: Markov time
    :param b: Number of negative samples used in the sampling algorithm.
    :return: PMI matrix.
    """

    x = stationary_distribution(M)
    X = np.diag(x)
    M_tau = np.linalg.matrix_power(M, tau)

    return np.log(X @ M_tau/b) - np.log(np.outer(x, x))

def preprocess_similarity_matrix(R):
    """
    Preprocess the similarity matrix.

    :param R: Similarity matrix.
    :return: Preprocessed similarity matrix.
    """

    # R = R.copy()

    # Replace nan with 0 and negative infinity with min value in the matrix.
    R[np.isnan(R)] = 0
    R[np.isinf(R)] = np.inf
    R[np.isinf(R)] = R.min()

    return R

def rescale_embeddings(u):
    """
    Rescale the embedding matrix by mean removal and variance scaling.

    :param u: Embeddings.
    :return: Rescaled embeddings.
    """
    shape = u.shape
    scaled = scale(u.flatten())
    return np.reshape(scaled, shape)

def postprocess_decomposition(u, s, v=None):
    """
    Postprocess the decomposed vectors and values into final embeddings.

    :param u: Eigenvectors (or left singular vectors)
    :param s: Eigenvalues (or singular values)
    :param v: Right singular vectors.
    :return: Embeddings.
    """

    dim = len(s)

    # Weight the vectors with square root of values.
    for i in range(dim):
        u[:, i] *= np.sqrt(s[i])
        if v is not None:
            v[:, i] *= np.sqrt(s[i])

    # Unify the sign of vectors for reproducible results.
    for i in range(dim):
        if u[0, i] < 0:
            u[:, i] *= -1
            if v is not None:
                v[:, i] *= -1

    # Rescale the embedding matrix.
    if v is not None:
        return rescale_embeddings(u), rescale_embeddings(v)
    else:
        return rescale_embeddings(u)
    
def get_node_labels(df):
    df.columns = ["node", "label"]
    node_labels = {}
    df_rows = df.shape[0]
    for i in range(df_rows):
        node = df["node"][i]
        label = df["label"][i]
        if node in node_labels:
            node_labels[node].append(label)
        else:
            node_labels[node] = [label]
    return node_labels

def get_category_list(node_labels):
    category = []
    n = len(node_labels)
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(n):
        for c in categories:
            if c in node_labels[i]:
                category.append(c)
                break
        else:
            category.append(0)
    return category

def transition_matrix_node_label_node(n, node_labels):
    arr_nln = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            common_count = len(set(node_labels[i]) & set(node_labels[j]))
            if common_count > 0:
                arr_nln[i, j] = common_count
                arr_nln[j, i] = common_count
    arr_sum = np.sum(arr_nln, axis=1)
    for i in range(n):
        if arr_sum[i] > 0:
            arr_nln[i] /= arr_sum[i]
    return arr_nln, arr_sum

def M_attr(n, M, node_labels, node_labels2, alpha, beta):
    arr_nln, arr_sum = transition_matrix_node_label_node(n, node_labels)
    if node_labels2 is None:
        for i in range(n):
            if arr_sum[i] > 0:
                M[i] = M[i]*alpha + arr_nln[i]*(1-alpha)
    else:
        arr_nln2, arr_sum2 = transition_matrix_node_label_node(n, node_labels2)
        for i in range(n):
            if arr_sum[i] > 0 and arr_sum2[i] == 0:
                M[i] = M[i]*alpha + arr_nln[i]*(1-alpha)
            elif arr_sum[i] == 0 and arr_sum2[i] > 0:
                M[i] = M[i]*alpha + arr_nln2[i]*(1-alpha)
            elif arr_sum[i] > 0 and arr_sum2[i] > 0:
                M[i] = M[i]*alpha + arr_nln[i]*(1-alpha)*beta + arr_nln2[i]*(1-alpha)*(1-beta)
    return M

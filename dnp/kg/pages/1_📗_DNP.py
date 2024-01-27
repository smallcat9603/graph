import streamlit as st
import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np

DATA = __file__.split("/")[-1].split(".")[0].split("_")[-1]
DATA_DIR = "https://raw.githubusercontent.com/smallcat9603/graph/main/dnp/kg/data/"
FILE_NODES = ["Article", "Noun", "Query"]
FILE_RELATIONSHIPS =["CONTAINS", "CORRELATES", "SIMILAR_JACCARD", "SIMILAR_OVERLAP", "SIMILAR_COSINE", "SIMILAR_FASTRP", "SIMILAR_NODE2VEC", "SIMILAR_HASHGNN"]

st.title(f"{DATA} Dataset")
st.info("This database includes 100 DNP newsreleases, and 4 Toppan newsreleases.")

st.title("Parameters")
form = st.form("parameters")
nphrase = form.slider("Number of nouns extracted from each article (50 if Offline is selected)", 1, 100, 50)
DATA_TYPE = form.radio("Data type", ["TXT", "URL"], horizontal=True, captions=["currently used only for dnp data", "parse html to retrive content"])
# offline opt: neo4j-admin database dump/load, require to stop neo4j server
DATA_LOAD = form.radio("Data load", ["Offline", "Semi-Online", "Online"], horizontal=True, captions=["load nodes and relationships from local (avoid to use gcp api, very fast)", "load nodes from local and create relationships during runtime (avoid to use gcp api, fast)", "create nodes and relationships during runtime (use gcp api, slow)"], index=0)
gcp_api_key = form.text_input('GCP API Key', type='password', placeholder='should not be empty for Online')
OUTPUT = form.radio("Output", ["Simple", "Verbose"], horizontal=True, captions=["user mode", "develeper mode (esp. for debug)"])

run_disabled = False
if "data" in st.session_state and st.session_state["data"] != DATA:
    run_disabled = True
    form.warning("Please 'Reset' the database status first before you 'Run'!", icon='⚠')
run = form.form_submit_button("Run", type="primary", disabled=run_disabled)
if not run and ("data" not in st.session_state or st.session_state["data"] != DATA):
    st.stop()
if run and DATA_LOAD == "Online" and gcp_api_key == "":
    st.stop()

DATA_URL = "" # input data
QUERY_DICT = {} # query dict {QUERY_NAME: QUERY_URL}
if DATA_TYPE == "TXT":
    DATA_URL = os.path.dirname(os.path.dirname(__file__)) + "/data/newsrelease_B-1-100_C-1-4/"
    QUERY_DICT["C-1"] = DATA_URL + "C-1.txt"
    QUERY_DICT["C-2"] = DATA_URL + "C-2.txt"
    QUERY_DICT["C-3"] = DATA_URL + "C-3.txt"
    QUERY_DICT["C-4"] = DATA_URL + "C-4.txt"
elif DATA_TYPE == "URL":
    DATA_URL = f"{DATA_DIR}articles.csv"
    QUERY_DICT["C-1"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_1.html"
    QUERY_DICT["C-2"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_2.html"
    QUERY_DICT["C-3"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_3.html"
    QUERY_DICT["C-4"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231003_1.html"

if OUTPUT == "Verbose":
    st.title("Data Source")
    st.write(DATA_URL)
    st.title("Query Dict")
    st.table(QUERY_DICT)

@st.cache_data
def cypher(query):
   return st.session_state["gds"].run_cypher(query)

query = """
CREATE CONSTRAINT id_unique IF NOT EXISTS 
For (a:Article) REQUIRE a.url IS UNIQUE;
"""
cypher(query)

##############################
### Import CSV ###
##############################

@st.cache_data
def import_graph_data():
    query = "CALL apoc.import.csv(["
    for idx, node in enumerate(FILE_NODES):
        query += f"{{fileName: '{DATA_DIR}{DATA}.nodes.{node}.csv', labels: ['{node}']}}, "
        if idx == len(FILE_NODES)-1:
            query = query[:-2] + "], ["
    for idx, relationship in enumerate(FILE_RELATIONSHIPS):
        query += f"{{fileName: '{DATA_DIR}{DATA}.relationships.{relationship}.csv', type: '{relationship}'}}, "
        if idx == len(FILE_RELATIONSHIPS)-1:
            query = query[:-2] + "], {})"
    result = cypher(query)
    return result

# convert string to value
@st.cache_data
def post_process():
    query = f"""
    MATCH (n) WHERE n.pr0 IS NOT NULL
    SET n.pr0 = toFloat(n.pr0)
    """
    cypher(query)
    query = f"""
    MATCH (n) WHERE n.pr1 IS NOT NULL
    SET n.pr1 = toFloat(n.pr1)
    SET n.pr2 = toFloat(n.pr2)
    SET n.pr3 = toFloat(n.pr3)
    """
    cypher(query)
    query = f"""
    MATCH (n) WHERE n.phrase IS NOT NULL
    SET n.salience = apoc.convert.fromJsonList(n.salience)
    """
    cypher(query)
    query = f"""
    MATCH ()-[r:CONTAINS]-() WHERE r.rank IS NOT NULL
    SET r.rank = toInteger(r.rank)
    SET r.weight = toInteger(r.weight)
    SET r.score = toFloat(r.score)
    """
    cypher(query)
    query = f"""
    MATCH ()-[r]-() WHERE type(r) =~ 'SIMILAR_.*'
    SET r.score = toFloat(r.score)
    """
    cypher(query)
    query = f"""
    MATCH (n) WHERE n.phrase IS NOT NULL
    SET n.phrase = replace(n.phrase, "[", "")
    SET n.phrase = replace(n.phrase, "]", "")
    SET n.phrase = split(n.phrase, ",")
    """
    cypher(query)    
    query = f"""
    MATCH ()-[r:CORRELATES]-() WHERE r.common IS NOT NULL
    SET r.common = replace(r.common, "[", "")
    SET r.common = replace(r.common, "]", "")
    WITH r, r.common AS common
    SET r.common = split(common, ",")
    """
    cypher(query) 

if DATA_LOAD == "Offline":
    result = import_graph_data()
    post_process()
    if OUTPUT == "Verbose":
        st.info("Importing nodes and relationships from csv files finished")
        st.write(result)

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
            cypher(query)
    else:
        query = f"""
        CALL apoc.periodic.iterate(
            "LOAD CSV WITH HEADERS FROM '{DATA_URL}' AS row
            RETURN row",
            "MERGE (a:Article {{name: row.id, url: row.url}})
            SET a.grp = CASE WHEN 'occupation' IN keys(row) THEN row.occupation ELSE null END
            SET a.grp1 = CASE WHEN 'nationality' IN keys(row) THEN row.nationality ELSE null END
            WITH a
            CALL apoc.load.html(a.url, {{
            title: 'title',
            h2: 'h2',
            body: 'body p'
            }})
            YIELD value
            WITH a,
                reduce(texts = '', n IN range(0, size(value.body)-1) | texts + ' ' + coalesce(value.body[n].text, '')) AS body,
                value.title[0].text AS title
            SET a.body = body, a.title = title",
            {{batchSize: 5, parallel: true}}
        )
        YIELD batches, total, timeTaken, committedOperations
        RETURN batches, total, timeTaken, committedOperations
        """
        cypher(query)

##############################
### set phrase and salience properties ###
##############################
  
progress_bar.progress(20, text="Set phrase and salience properties...")

if DATA_LOAD == "Semi-Online":
    query = f"""
    LOAD CSV WITH HEADERS FROM "{DATA_DIR}{DATA}.csv" AS row
    WITH row
    WHERE row._labels = ":Article"
    MATCH (a:Article {{name: row.name}}) WHERE a.processed IS NULL
    SET a.processed = true
    SET a.phrase = apoc.convert.fromJsonList(row.phrase)
    SET a.salience = apoc.convert.fromJsonList(row.salience)
    RETURN COUNT(a) AS Processed
    """
    result = cypher(query)
    if OUTPUT == "Verbose":
        st.write(result)
elif DATA_LOAD == "Online":
    query = f"""
    CALL apoc.periodic.iterate(
    "MATCH (a:Article)
    WHERE a.processed IS NULL
    RETURN a",
    "CALL apoc.nlp.gcp.entities.stream([item in $_batch | item.a], {{
        nodeProperty: 'body',
        key: '{gcp_api_key}'
    }})
    YIELD node, value
    SET node.processed = true
    WITH node, value
    UNWIND value.entities AS entity
    SET node.phrase = coalesce(node.phrase, []) + entity['name']
    SET node.salience = coalesce(node.salience, []) + entity['salience']",
    {{batchMode: "BATCH_SINGLE", batchSize: 10}})
    YIELD batches, total, timeTaken, committedOperations
    RETURN batches, total, timeTaken, committedOperations
    """
    result = cypher(query)
    if OUTPUT == "Verbose":
        st.write(result)

##############################
### create noun-url relationships ###
##############################

progress_bar.progress(30, text="Create noun-url relationships...")

if DATA_LOAD != "Offline":
    query = f"""
    MATCH (a:Article)
    WHERE a.processed IS NOT NULL
    FOREACH (word IN a.phrase[0..{nphrase}] |
    MERGE (n:Noun {{name: word}})
    MERGE (a)-[r:CONTAINS]-(n)
    SET r.rank = apoc.coll.indexOf(a.phrase, word) + 1
    SET r.score = a.salience[apoc.coll.indexOf(a.phrase, word)]
    SET r.weight = {nphrase} - apoc.coll.indexOf(a.phrase, word)
    )
    """
    cypher(query)

##############################
### query ###
##############################

progress_bar.progress(40, text="Create query nodes...")

if DATA_LOAD != "Offline":
    if DATA_TYPE == "TXT":
        for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
            content = ""
            with open(QUERY_URL, 'r') as f:
                content = f.read()
                content = re.sub('\n+', ' ', content)
            query = f"""
            MERGE (q:Query {{ name: "{QUERY_NAME}", url: "{QUERY_URL}", body: "{content}" }})
            """
            cypher(query)
    else:
        for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
            query = f"""
            MERGE (q:Query {{name: "{QUERY_NAME}", url: "{QUERY_URL}"}})
            WITH q
            CALL apoc.load.html(q.url, {{
            title: "title",
            h2: "h2",
            body: "body p"
            }})
            YIELD value
            WITH q,
                reduce(texts = "", n IN range(0, size(value.body)-1) | texts + " " + coalesce(value.body[n].text, "")) AS body,
                value.title[0].text AS title
            SET q.body = body, q.title = title
            RETURN q.title, q.body
            """
            cypher(query)

progress_bar.progress(50, text="Set phrase and salience properties (Query)...")
    
# set phrase and salience properties (Query)
if DATA_LOAD == "Semi-Online":
    query = f"""
    LOAD CSV WITH HEADERS FROM "{DATA_DIR}{DATA}.csv" AS row
    WITH row
    WHERE row._labels = ":Query"
    MATCH (q:Query {{name: row.name}})
    SET q.processed = true
    SET q.phrase = apoc.convert.fromJsonList(row.phrase)
    SET q.salience = apoc.convert.fromJsonList(row.salience)
    """
    cypher(query)
elif DATA_LOAD == "Online":
    query = f"""
    MATCH (q:Query)
    CALL apoc.nlp.gcp.entities.stream(q, {{
    nodeProperty: 'body',
    key: '{gcp_api_key}'
    }})
    YIELD node, value
    SET node.processed = true
    WITH node, value
    UNWIND value.entities AS entity
    SET node.phrase = coalesce(node.phrase, []) + entity['name']
    SET node.salience = coalesce(node.salience, []) + entity['salience']
    """
    cypher(query)

progress_bar.progress(60, text="Create noun-article relationships (Query)...")

# create noun-article relationships (Query)
if DATA_LOAD != "Offline":
    query = f"""
    MATCH (q:Query)
    WHERE q.processed IS NOT NULL
    FOREACH (word IN q.phrase[0..{nphrase}] |
    MERGE (n:Noun {{name: word}})
    MERGE (q)-[r:CONTAINS]-(n)
    SET r.rank = apoc.coll.indexOf(q.phrase, word) + 1
    SET r.score = q.salience[apoc.coll.indexOf(q.phrase, word)]
    SET r.weight = {nphrase} - apoc.coll.indexOf(q.phrase, word)
    )
    """
    cypher(query)

##############################
### create article-article relationships ###
##############################

progress_bar.progress(70, text="Create article-article relationships...")

if DATA_LOAD != "Offline":
    query = f"""
    MATCH (a1:Article), (a2:Article)
    WHERE a1 <> a2 AND any(x IN a1.phrase[0..{nphrase}] WHERE x IN a2.phrase[0..{nphrase}])
    MERGE (a1)-[r:CORRELATES]-(a2)
    SET r.common = [x IN a1.phrase[0..{nphrase}] WHERE x IN a2.phrase[0..{nphrase}]]
    """
    cypher(query)

    #query
    query = f"""
    MATCH (q:Query), (a:Article)
    WHERE any(x IN q.phrase[0..{nphrase}] WHERE x IN a.phrase[0..{nphrase}])
    MERGE (q)-[r:CORRELATES]-(a)
    SET r.common = [x IN q.phrase[0..{nphrase}] WHERE x IN a.phrase[0..{nphrase}]]
    """
    cypher(query)

##############################
### project graph to memory ###
##############################

progress_bar.progress(80, text="Project graph to memory...")

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

##############################
### graph statistics ###
##############################

if OUTPUT == "Verbose":

    st.divider()
    st.title("Graph Statistics (project)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("# Nodes", str(G.node_count()))
    col2.metric("# Edges", str(G.relationship_count()))
    col3.metric("Density", str(G.density()))
    col4.metric("Memory", str(G.memory_usage()))

##############################
### node similarity (JACCARD) ###
##############################

@st.cache_data
def write_nodesimilarity_jaccard():

    result = st.session_state["gds"].nodeSimilarity.filtered.write(
        G,
        similarityMetric='JACCARD', # default
        writeRelationshipType='SIMILAR_JACCARD',
        writeProperty='score',
        relationshipWeightProperty="weight",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
        topK=100,
    )
    if OUTPUT == "Verbose":
        st.title("node similarity (JACCARD)")
        st.write(f"Relationships produced: {result['relationshipsWritten']}")
        st.write(f"Nodes compared: {result['nodesCompared']}")
        st.write(f"Mean similarity: {result['similarityDistribution']['mean']}")

if DATA_LOAD != "Offline":
    write_nodesimilarity_jaccard()

##############################
### node similarity (OVERLAP) ###
##############################

@st.cache_data
def write_nodesimilarity_overlap():

    result = st.session_state["gds"].nodeSimilarity.filtered.write(
        G,
        similarityMetric='OVERLAP',
        writeRelationshipType='SIMILAR_OVERLAP',
        writeProperty='score',
        relationshipWeightProperty="weight",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
        topK=100,
    )
    if OUTPUT == "Verbose":
        st.title("node similarity (OVERLAP)")
        st.write(f"Relationships produced: {result['relationshipsWritten']}")
        st.write(f"Nodes compared: {result['nodesCompared']}")
        st.write(f"Mean similarity: {result['similarityDistribution']['mean']}")

if DATA_LOAD != "Offline":
    write_nodesimilarity_overlap()

##############################
### node similarity (COSINE) ###
##############################

@st.cache_data
def write_nodesimilarity_cosine():

    result = st.session_state["gds"].nodeSimilarity.filtered.write(
        G,
        similarityMetric='COSINE',
        writeRelationshipType='SIMILAR_COSINE',
        writeProperty='score',
        relationshipWeightProperty="weight",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
        topK=100,
    )
    if OUTPUT == "Verbose":
        st.title("node similarity (COSINE)")
        st.write(f"Relationships produced: {result['relationshipsWritten']}")
        st.write(f"Nodes compared: {result['nodesCompared']}")
        st.write(f"Mean similarity: {result['similarityDistribution']['mean']}")

if DATA_LOAD != "Offline":
    write_nodesimilarity_cosine()

##############################
### ppr (personalized pagerank) ###
##############################

@st.cache_data
def write_nodesimilarity_ppr():

    for idx, name in enumerate(list(QUERY_DICT.keys())):
        nodeid = st.session_state["gds"].find_node_id(labels=["Query"], properties={"name": name})
        result = st.session_state["gds"].pageRank.write(
            G,
            writeProperty="pr"+str(idx),
            maxIterations=20,
            dampingFactor=0.85,
            relationshipWeightProperty='weight',
            sourceNodes=[nodeid]
        )   
        if OUTPUT == "Verbose":
            st.write(f"Node properties written: {result['nodePropertiesWritten']}")
            st.write(f"Mean: {result['centralityDistribution']['mean']}")

if DATA_LOAD != "Offline":
    write_nodesimilarity_ppr()

##############################
### 1. node embedding ###
##############################

progress_bar.progress(90, text="Node embedding...")

@st.cache_data
def node_embedding():

    # fastrp
    result = st.session_state["gds"].fastRP.stream(
        G,
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight",
        iterationWeights=[1, 1, 1],
    )

    # node2vec
    result = st.session_state["gds"].node2vec.stream(
        G,
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight",
        iterations=3,
    )

    # hashgnn
    result = st.session_state["gds"].beta.hashgnn.stream(
        G,
        iterations = 3,
        embeddingDensity = 8,
        generateFeatures = {"dimension": 16, "densityLevel": 1},
        randomSeed = 42,
    )

    if OUTPUT == "Verbose":
        st.write(f"Embedding vectors: {result['embedding']}")

    # fastrp
    result = st.session_state["gds"].fastRP.mutate(
        G,
        mutateProperty="embedding_fastrp",
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight", # each relationship should have
        iterationWeights=[1, 1, 1],
    )

    # node2vec
    result = st.session_state["gds"].node2vec.mutate(
        G,
        mutateProperty="embedding_node2vec",
        randomSeed=42,
        embeddingDimension=16,
        relationshipWeightProperty="weight",
        iterations=3,
    )

    # hashgnn
    result = st.session_state["gds"].beta.hashgnn.mutate(
        G,
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

    if OUTPUT == "Verbose":
        st.title("1. node embedding")
        st.write(f"Number of embedding vectors produced: {result['nodePropertiesWritten']}")

if DATA_LOAD != "Offline":
    node_embedding()

##############################
### 2. kNN ###
##############################

@st.cache_data
def kNN():

    # fastrp
    result = st.session_state["gds"].knn.filtered.write(
        G,
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
        G,
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
        G,
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

    if OUTPUT == "Verbose":
        st.title("2. kNN")
        st.write(f"Relationships produced: {result['relationshipsWritten']}")
        st.write(f"Nodes compared: {result['nodesCompared']}")
        st.write(f"Mean similarity: {result['similarityDistribution']['mean']}")

if DATA_LOAD != "Offline":
    kNN()

##############################
### evaluate (node embedding + knn) ###
##############################

# fastrp
query = """
MATCH (q:Query)-[r:SIMILAR_FASTRP]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""

if OUTPUT == "Verbose":
    st.title("evaluate (fastrp)")
    st.write(cypher(query))

# node2vec
query = """
MATCH (q:Query)-[r:SIMILAR_NODE2VEC]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""

if OUTPUT == "Verbose":
    st.title("evaluate (node2vec)")
    st.write(cypher(query))

# hashgnn
query = """
MATCH (q:Query)-[r:SIMILAR_HASHGNN]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""

if OUTPUT == "Verbose":
    st.title("evaluate (hashgnn)")
    st.write(cypher(query))

##############################
### export to csv in import/ ###
##############################

st.divider()
st.title("Graph Statistics")
query = f"""
CALL apoc.meta.stats()
YIELD nodeCount, relCount, labels, relTypesCount
RETURN nodeCount, relCount, labels, relTypesCount
"""
result = cypher(query)

col1, col2 = st.columns(2)
col1.metric("# Nodes", result["nodeCount"][0])
col2.metric("# Edges", result["relCount"][0])

col1, col2 = st.columns(2)
with col1.expander("Node Labels"):
    st.table(result["labels"][0])
with col2.expander("Relationship Types"):
    st.table(result["relTypesCount"][0])

if OUTPUT == "Verbose":
    st.caption("Save graph data including nodes and edges into csv files")
    if st.button("Save graph data (.csv)"):
        # no bulkImport: all in one
        # use bulkImport to generate multiple files categorized by node label and relationship type
        query = f"""
        CALL apoc.export.csv.all("{DATA}.csv", {{}}) 
        """
        result_allinone = cypher(query)
        query = f"""
        CALL apoc.export.csv.all("{DATA}.csv", {{bulkImport: true}}) 
        """
        result_bulkimport = cypher(query)

        st.write(result_allinone)
        # st.write(result_bulkimport)
    
progress_bar.progress(100, text="Finished.")
end_time = time.perf_counter()
execution_time_ms = (end_time - start_time) * 1000
container_status.success(f"Loading finished: {execution_time_ms:.1f} ms. Graph data can be queried.")

st.session_state["data"] = DATA

##############################
### interaction ###
##############################

@st.cache_data
def plot_similarity(query_node, similarity_method, limit):
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

st.divider()
st.title("UI Interaction")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Node Similarity", "Multiple Queries", "Related Articles", "Common Keywords", "Naive by Rank", "Naive by Salience"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        query_node = st.selectbox("Query node", ("C-1", "C-2", "C-3", "C-4"))
    with col2:
        similarity_method = st.selectbox("Similarity method", ("JACCARD", "OVERLAP", "COSINE", "PPR"))
    with col3:
        limit = st.selectbox("Limit", ("5", "10", "15", "20"))
    st.write("The top-" + limit + " similar nodes for query " + query_node + " are ranked as follows (" + similarity_method + ")")
    if similarity_method == "PPR":
        query = f"""
        MATCH (q:Query)-[r:CORRELATES]-(a:Article) WHERE q.name = "{query_node}"
        RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.pr{str(int(query_node.split("-")[-1])-1)} AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """ 
    else:
        query = f"""
        MATCH (q:Query)-[r:SIMILAR_{similarity_method}]-(a:Article) WHERE q.name = "{query_node}"
        RETURN q.name AS Query, a.name AS Article, a.url AS URL, r.score AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """
    if OUTPUT == "Verbose":
        st.code(query)    
    result = cypher(query)
    tab01, tab02 = st.tabs(["Chart", "Table"])
    with tab01:
        plot_similarity(query_node, similarity_method, limit)
    with tab02:
        st.write(result)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        query_nodes = st.multiselect("Query node", ["C-1", "C-2", "C-3", "C-4"], ["C-1", "C-2"])
    with col2:
        similarity_method = st.selectbox("Similarity method", ("JACCARD", "OVERLAP", "COSINE", "PPR"), key="sm")
    with col3:
        limit = st.selectbox("Limit", ("5", "10", "15", "20"), key="lim")
    st.write("The top-" + limit + " similar nodes for queries " + ', '.join(query_nodes) + " are ranked as follows (" + similarity_method + ")")
    ppr_attr = ["pr" + str(int(item.replace("C-", ""))-1) for item in query_nodes]
    if similarity_method == "PPR":
        query = f"""
        MATCH (q:Query)-[r:CORRELATES]-(a:Article) WHERE q.name IN {query_nodes}
        RETURN COLLECT(q.name) AS Query, a.name AS Article, REDUCE(s = 0, pr IN {ppr_attr} | s + a[pr]) AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """ 
    else:
        query = f"""
        MATCH (q:Query)-[r:SIMILAR_{similarity_method}]-(a:Article) WHERE q.name IN {query_nodes}
        RETURN COLLECT(q.name) AS Query, a.name AS Article, SUM(r.score) AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """
    if OUTPUT == "Verbose":
        st.code(query)    
    result = cypher(query)
    tab01, tab02 = st.tabs(["Chart", "Table"])
    with tab01:
        plot_similarity(', '.join(query_nodes), similarity_method, limit)
    with tab02:
        st.write(result)

with tab3:
    noun = st.text_input("Keyword", "環境")
    query = f"""
    MATCH (n:Noun)-[]-(a:Article) WHERE n.name CONTAINS "{noun}"
    WITH DISTINCT a AS distinctArticle, n
    RETURN n.name AS Keyword, COUNT(distinctArticle) AS articleCount, COLLECT(distinctArticle.name) AS articles
    ORDER BY articleCount DESC
    """
    if OUTPUT == "Verbose":
        st.code(query)    
    result = cypher(query)
    st.write(result)

with tab4:
    query = f"""
    MATCH (n:Noun)-[]-(a:Article)
    RETURN n.name AS Keyword, COUNT(a) AS articleCount, COLLECT(a.name) AS articles
    ORDER BY articleCount DESC
    """
    if OUTPUT == "Verbose":
        st.code(query)    
    result = cypher(query)
    st.write(result)

with tab5:
    query = """
    MATCH (q:Query)-[r:CONTAINS]-(n:Noun)-[c:CONTAINS]-(a:Article)
    RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, collect(n.name) AS Common, SUM((1.0/r.rank)*(1.0/c.rank)) AS Similarity 
    ORDER BY Query, Similarity DESC
    LIMIT 10
    """
    if OUTPUT == "Verbose":
        st.code(query)
    result = cypher(query)
    st.write(result)

with tab6:
    query = """
    MATCH (q:Query)-[r:CORRELATES]-(a:Article)
    WITH q, r, a, reduce(s = 0.0, word IN r.common | 
    s + q.salience[apoc.coll.indexOf(q.phrase, word)] + a.salience[apoc.coll.indexOf(a.phrase, word)]) AS Similarity
    RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.common, Similarity 
    ORDER BY Query, Similarity DESC
    LIMIT 10
    """
    if OUTPUT == "Verbose":
        st.code(query)
    result = cypher(query)
    st.write(result)

st.divider()
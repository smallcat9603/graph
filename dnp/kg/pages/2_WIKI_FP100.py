import streamlit as st
import pandas as pd
import os

st.header("Parameters")
form = st.form("parameters")
nphrase = form.slider("Number of nouns extracted from each article", 1, 100, 50)
DATA_TYPE = form.radio("Data type", ["URL"], horizontal=True, captions=["parse html to retrive content"])
# offline opt: neo4j-admin database dump/load, require to stop neo4j server
DATA_LOAD = form.radio("Data load", ["Offline", "Semi-Online", "Online"], horizontal=True, captions=["load nodes and relationships from local (avoid to use gcp api, very fast)", "load nodes from local and create relationships during runtime (avoid to use gcp api, fast)", "create nodes and relationships during runtime (use gcp api, slow)"], index=0)
OUTPUT = form.radio("Output", ["Simple", "Verbose"], horizontal=True, captions=["user mode", "develeper mode (esp. for debug)"])

run = form.form_submit_button("Run")
if not run:
    st.stop()

DATA_URL = "" # input data
QUERY_DICT = {} # query dict {QUERY_NAME: QUERY_URL}
DATA_URL = "https://raw.githubusercontent.com/smallcat9603/graph/main/dnp/kg/data/wikidata_footballplayer_100.csv"
QUERY_DICT["Thierry Henry"] = "https://en.wikipedia.org/wiki/Thierry_Henry"

FILE_NAME = __file__.split("/")[-1].split(".")[0].split("_")[-1]

if OUTPUT == "Verbose":
    st.header("data url")
    st.write(DATA_URL)
    st.header("query dict")
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

query = """
CALL dbms.listConfig() YIELD name, value
WHERE name = 'server.directories.import'
RETURN value AS importFolderPath
"""
result = cypher(query)
importFolderPath = result["importFolderPath"][0]

filenames_nodes = []
filenames_relationships =[]
for filename in os.listdir(importFolderPath):
    if filename.startswith(FILE_NAME+".nodes.") and filename.endswith(".csv"):
        filenames_nodes.append(filename)
    if filename.startswith(FILE_NAME+".relationships.") and filename.endswith(".csv"):
        filenames_relationships.append(filename)

def import_graph_data():
    query = "CALL apoc.import.csv(["
    for idx, filename in enumerate(filenames_nodes):
        if idx < len(filenames_nodes)-1:
            query += f"{{fileName: 'file:/{filename}', labels: ['{filename.split('.')[-2]}']}}, "
        else:
            query += f"{{fileName: 'file:/{filename}', labels: ['{filename.split('.')[-2]}']}}], ["
    for idx, filename in enumerate(filenames_relationships):
        if idx < len(filenames_relationships)-1:
            query += f"{{fileName: 'file:/{filename}', type: '{filename.split('.')[-2]}'}}, "
        else:
            query += f"{{fileName: 'file:/{filename}', type: '{filename.split('.')[-2]}'}}], {{}})"
    result = cypher(query)
    return result

# convert string to value
# TODO: n.phrase not converted to stringlist via MATCH (n) WHERE n.phrase IS NOT NULL
# TODO: r.common not converted to stringlist via MATCH ()-[r:CORRELATES]-() WHERE r.common IS NOT NULL
def post_process():
    query = f"""
    MATCH (n) WHERE n.pr0 IS NOT NULL
    SET n.pr0 = toFloat(n.pr0)
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

if DATA_LOAD == "Offline":
    result = import_graph_data()
    post_process()
    if OUTPUT == "Verbose":
        st.info("Importing nodes and relationships from csv files finished")
        st.write(result)

##############################
### Create Article-[Noun]-Article Graph ###
##############################

st.header("Progress")
progress_bar = st.progress(0, text="Initialize...")

##############################
### create url nodes (article, person, ...) ###
##############################

progress_bar.progress(10, text="Create url nodes...")

if DATA_LOAD != "Offline":

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
    LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/smallcat9603/graph/main/dnp/kg/data/{FILE_NAME}.csv" AS row
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
        key: '{st.secrets["GCP_API_KEY"]}'
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
    LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/smallcat9603/graph/main/dnp/kg/data/{FILE_NAME}.csv" AS row
    WITH row
    WHERE row._labels = ":Query"
    MATCH (q:Query {{name: row.name}})
    SET q.processed = true
    SET q.phrase = apoc.convert.fromJsonList(row.phrase)
    SET q.salience = apoc.convert.fromJsonList(row.salience)
    """
elif DATA_LOAD == "Online":
    query = f"""
    MATCH (q:Query)
    CALL apoc.nlp.gcp.entities.stream(q, {{
    nodeProperty: 'body',
    key: '{st.secrets["GCP_API_KEY"]}'
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

if DATA_LOAD != "Offline":

    # create noun-article relationships (Query)
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
### evaluate (naive by rank) ###
##############################

query = """
MATCH (q:Query)-[r:CONTAINS]-(n:Noun)-[c:CONTAINS]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, collect(n.name) AS Common, SUM((1.0/r.rank)*(1.0/c.rank)) AS Similarity 
ORDER BY Query, Similarity DESC
LIMIT 10
"""

if OUTPUT == "Verbose":
    st.header("evaluate (naive by rank)")
    st.write(cypher(query))

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
### evaluate (still naive by salience) ###
##############################

query = """
MATCH (q:Query)-[r:CORRELATES]-(a:Article)
WITH q, r, a, reduce(s = 0.0, word IN r.common | 
s + q.salience[apoc.coll.indexOf(q.phrase, word)] + a.salience[apoc.coll.indexOf(a.phrase, word)]) AS Similarity
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.common, Similarity 
ORDER BY Query, Similarity DESC
LIMIT 10
"""

if OUTPUT == "Verbose":
    st.header("evaluate (still naive by salience)")
    st.write(cypher(query))

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
st.header("project graph to memory")
st.write(f"The projection took {result['projectMillis']} ms")
st.write(f"Graph '{G.name()}' node count: {G.node_count()}")
st.write(f"Graph '{G.name()}' node labels: {G.node_labels()}")
st.write(f"Graph '{G.name()}' relationship count: {G.relationship_count()}")
st.write(f"Graph '{G.name()}' degree distribution: {G.degree_distribution()}")
st.write(f"Graph '{G.name()}' density: {G.density()}")
st.write(f"Graph '{G.name()}' size in bytes: {G.size_in_bytes()}")
st.write(f"Graph '{G.name()}' memory_usage: {G.memory_usage()}")

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
    st.header("node similarity (JACCARD)")
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
    st.header("node similarity (OVERLAP)")
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
    st.header("node similarity (COSINE)")
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

    st.header("ppr (personalized pagerank)")
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

    # st.write(f"Embedding vectors: {result['embedding']}")

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

    st.header("1. node embedding")
    st.write(f"Number of embedding vectors produced: {result['nodePropertiesWritten']}")

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

    st.header("2. kNN")
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
st.header("evaluate (fastrp)")
st.write(cypher(query))

# node2vec
query = """
MATCH (q:Query)-[r:SIMILAR_NODE2VEC]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""
st.header("evaluate (node2vec)")
st.write(cypher(query))

# hashgnn
query = """
MATCH (q:Query)-[r:SIMILAR_HASHGNN]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""
st.header("evaluate (hashgnn)")
st.write(cypher(query))

##############################
### export to csv in import/ ###
##############################

# no bulkImport: all in one
# use bulkImport to generate multiple files categorized by node label and relationship type
def save_graph_data():
    query = f"""
    CALL apoc.export.csv.all("{FILE_NAME}.csv", {{}}) 
    """
    result_allinone = cypher(query)
    query = f"""
    CALL apoc.export.csv.all("{FILE_NAME}.csv", {{bulkImport: true}}) 
    """
    result_bulkimport = cypher(query)
    if OUTPUT == "Verbose":
        st.write(cypher(result_allinone))
        st.write(cypher(result_bulkimport))

st.button("Save graph data", on_click=save_graph_data) 

##############################
### interaction ###
##############################

st.header("UI Interaction")

tab1, tab2, tab3 = st.tabs(["Node Similarity", "Related Articles", "Common Keywords"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        query_node = st.selectbox("Query node", ("Thierry Henry", "C-2", "C-3", "C-4"))
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
        df = pd.DataFrame(
            data=list(result["Similarity"]),
            columns=["Similarity Score"],
            index=result["Article"],
        )
        st.bar_chart(df)
    with tab02:
        st.write(result)

with tab2:
    noun = st.text_input("Keyword", "football")
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

with tab3:
    query = f"""
    MATCH (n:Noun)-[]-(a:Article)
    RETURN n.name AS Keyword, COUNT(a) AS articleCount, COLLECT(a.name) AS articles
    ORDER BY articleCount DESC
    """
    if OUTPUT == "Verbose":
        st.code(query)    
    result = cypher(query)
    st.write(result)

progress_bar.progress(100, text="Finished. Graph data can be queried.")

st.session_state["data"] = "WIKI_FP100"
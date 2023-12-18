from graphdatascience import GraphDataScience
import sys, os
import re
import streamlit as st

##############################
### neo4j desktop v5.11.0 ###
##############################

if 'reboot' not in st.session_state:
   st.session_state["reboot"] = False
if st.session_state["reboot"] == True:
   sys.exit(0)

st.sidebar.title(__file__.split("/")[-1])

host = "bolt://localhost:7687"
user = "neo4j"
password= "j4oenj4oen"

gds = GraphDataScience(host, auth=(user, password))
st.sidebar.header("gds version")
st.sidebar.write(gds.version())

graph_name = "testgraph" # project graph name

##############################
### parameters ###
##############################

st.sidebar.header("parameters")
KEY = "AIzaSyAPQNUpCCFrsJhX2A-CgvOG4fDWlxuA8ec" # api key
nphrase = st.sidebar.slider("Number of nouns extracted from each article", 1, 100, 50)
DATA_CLASS = st.sidebar.radio("Data class", ["DNP", "WIKI_FP100", "WIKI_P100"])
DATA_TYPE = st.sidebar.radio("Data type (currently txt is used for dnp data)", ["TXT", "URL"])
DATA_URL = "" # input data
QUERY_DICT = {} # query dict {QUERY_NAME: QUERY_URL}
if DATA_CLASS == "DNP":
    if DATA_TYPE == "TXT":
        DATA_URL = os.path.dirname(__file__) + "/data/newsrelease_B-1-100_C-1-4/"
        QUERY_DICT["C-1"] = DATA_URL + "C-1.txt"
        QUERY_DICT["C-2"] = DATA_URL + "C-2.txt"
        QUERY_DICT["C-3"] = DATA_URL + "C-3.txt"
        QUERY_DICT["C-4"] = DATA_URL + "C-4.txt"
    elif DATA_TYPE == "URL":
        DATA_URL = "https://raw.githubusercontent.com/smallcat9603/graph/main/dnp/kg/data/articles.csv"
        QUERY_DICT["C-1"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_1.html"
        QUERY_DICT["C-2"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_2.html"
        QUERY_DICT["C-3"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_3.html"
        QUERY_DICT["C-4"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231003_1.html"
elif DATA_CLASS == "WIKI_FP100":
    DATA_URL = "https://raw.githubusercontent.com/smallcat9603/graph/main/dnp/kg/data/wikidata_footballplayer_100.csv"
    QUERY_DICT["Thierry Henry"] = "https://en.wikipedia.org/wiki/Thierry_Henry"
elif DATA_CLASS == "WIKI_P100":
    DATA_URL = "https://raw.githubusercontent.com/smallcat9603/graph/main/dnp/kg/data/wikidata_persons_100.csv"  
    QUERY_DICT["Joe Biden"] = "https://en.wikipedia.org/wiki/Joe_Biden"
else:
    print("DATA ERROR")
    sys.exit(1)

st.sidebar.header("data url")
st.sidebar.write(DATA_URL)
st.header("query dict")
st.write(QUERY_DICT)

query = """
CREATE CONSTRAINT id_unique IF NOT EXISTS 
For (a:Article) REQUIRE a.url IS UNIQUE;
"""
gds.run_cypher(query)

##############################
### Create Article-[Noun]-Article Graph ###
##############################

##############################
### create url nodes (article, person, ...) ###
##############################

if DATA_CLASS == "DNP" and DATA_TYPE == "TXT":
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
    gds.run_cypher(query)
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
  gds.run_cypher(query)

##############################
### set phrase and salience properties ###
##############################

query = f"""
CALL apoc.periodic.iterate(
  "MATCH (a:Article)
   WHERE a.processed IS NULL
   RETURN a",
  "CALL apoc.nlp.gcp.entities.stream([item in $_batch | item.a], {{
     nodeProperty: 'body',
     key: '{KEY}'
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
st.header("set phrase and salience properties")
st.write(gds.run_cypher(query))

##############################
### create noun-url relationships ###
##############################

query = """
MATCH (a:Article)
WHERE a.processed IS NOT NULL
FOREACH (word IN a.phrase[0..$nphrase] |
  MERGE (n:Noun {name: word})
  MERGE (a)-[r:CONTAINS]-(n)
  SET r.rank = apoc.coll.indexOf(a.phrase, word) + 1
  SET r.score = a.salience[apoc.coll.indexOf(a.phrase, word)]
  SET r.weight = $nphrase - apoc.coll.indexOf(a.phrase, word)
)
"""
gds.run_cypher(query, {'nphrase': nphrase})

##############################
### query ###
##############################

if DATA_CLASS == "DNP" and DATA_TYPE == "TXT":
  for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
    content = ""
    with open(QUERY_URL, 'r') as f:
      content = f.read()
      content = re.sub('\n+', ' ', content)
    query = f"""
    MERGE (q:Query {{ name: "{QUERY_NAME}", url: "{QUERY_URL}", body: "{content}" }})
    """
    gds.run_cypher(query)
else:
  for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
    query = """
    MERGE (q:Query {name: $name, url: $url})
    WITH q
    CALL apoc.load.html(i.url, {
    title: "title",
    h2: "h2",
    body: "body p"
    })
    YIELD value
    WITH q,
        reduce(texts = "", n IN range(0, size(value.body)-1) | texts + " " + coalesce(value.body[n].text, "")) AS body,
        value.title[0].text AS title
    SET q.body = body, q.title = title
    RETURN q.title, q.body
    """
    gds.run_cypher(query, {"name": QUERY_NAME, "url": QUERY_URL})
    
# set phrase and salience properties (Query)
query = f"""
MATCH (q:Query)
CALL apoc.nlp.gcp.entities.stream(q, {{
 nodeProperty: 'body',
 key: '{KEY}'
}})
YIELD node, value
SET node.processed = true
WITH node, value
UNWIND value.entities AS entity
SET node.phrase = coalesce(node.phrase, []) + entity['name']
SET node.salience = coalesce(node.salience, []) + entity['salience']
"""
gds.run_cypher(query)

# create noun-article relationships (Query)
query = """
MATCH (q:Query)
WHERE q.processed IS NOT NULL
FOREACH (word IN q.phrase[0..$nphrase] |
  MERGE (n:Noun {name: word})
  MERGE (q)-[r:CONTAINS]-(n)
  SET r.rank = apoc.coll.indexOf(q.phrase, word) + 1
  SET r.score = q.salience[apoc.coll.indexOf(q.phrase, word)]
  SET r.weight = $nphrase - apoc.coll.indexOf(q.phrase, word)
)
"""
gds.run_cypher(query, {'nphrase': nphrase})

##############################
### evaluate (naive by rank) ###
##############################

query = """
MATCH (q:Query)-[r:CONTAINS]-(n:Noun)-[c:CONTAINS]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, collect(n.name) AS Common, SUM((1.0/r.rank)*(1.0/c.rank)) AS Similarity 
ORDER BY Query, Similarity DESC
LIMIT 10
"""
st.header("evaluate (naive by rank)")
st.write(gds.run_cypher(query))

##############################
### create article-article relationships ###
##############################

query = """
MATCH (a1:Article), (a2:Article)
WHERE a1 <> a2 AND any(x IN a1.phrase[0..$nphrase] WHERE x IN a2.phrase[0..$nphrase])
MERGE (a1)-[r:CORRELATES]-(a2)
SET r.common = [x IN a1.phrase[0..$nphrase] WHERE x IN a2.phrase[0..$nphrase]]
"""
gds.run_cypher(query, {'nphrase': nphrase})

#query
query = """
MATCH (q:Query), (a:Article)
WHERE any(x IN q.phrase[0..$nphrase] WHERE x IN a.phrase[0..$nphrase])
MERGE (q)-[r:CORRELATES]-(a)
SET r.common = [x IN q.phrase[0..$nphrase] WHERE x IN a.phrase[0..$nphrase]]
"""
gds.run_cypher(query, {'nphrase': nphrase})

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
st.header("evaluate (still naive by salience)")
st.write(gds.run_cypher(query))

##############################
### project graph to memory ###
##############################

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

exists_result = gds.graph.exists(graph_name)
if exists_result["exists"]:
    G = gds.graph.get(graph_name)
    G.drop()
G, result = gds.graph.project(graph_name, node_projection, relationship_projection)
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

result = gds.nodeSimilarity.filtered.write(
    G,
    similarityMetric='JACCARD', # default
    writeRelationshipType='SIMILAR_JACCARD',
    writeProperty='score',
    relationshipWeightProperty="weight",
    sourceNodeFilter="Query",
    targetNodeFilter="Article",
)
st.header("node similarity (JACCARD)")
st.write(f"Relationships produced: {result['relationshipsWritten']}")
st.write(f"Nodes compared: {result['nodesCompared']}")
st.write(f"Mean similarity: {result['similarityDistribution']['mean']}")

##############################
### evaluate (jaccard similarity) ###
##############################

query = """
MATCH (q:Query)-[r:SIMILAR_JACCARD]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""
st.header("evaluate (jaccard similarity)")
st.write(gds.run_cypher(query))

##############################
### node similarity (OVERLAP) ###
##############################

result = gds.nodeSimilarity.filtered.write(
    G,
    similarityMetric='OVERLAP',
    writeRelationshipType='SIMILAR_OVERLAP',
    writeProperty='score',
    relationshipWeightProperty="weight",
    sourceNodeFilter="Query",
    targetNodeFilter="Article",
)
st.header("node similarity (OVERLAP)")
st.write(f"Relationships produced: {result['relationshipsWritten']}")
st.write(f"Nodes compared: {result['nodesCompared']}")
st.write(f"Mean similarity: {result['similarityDistribution']['mean']}")

##############################
### evaluate (overlap similarity) ###
##############################

query = """
MATCH (q:Query)-[r:SIMILAR_OVERLAP]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""
st.header("evaluate (overlap similarity)")
st.write(gds.run_cypher(query))

##############################
### node similarity (COSINE) ###
##############################

result = gds.nodeSimilarity.filtered.write(
    G,
    similarityMetric='COSINE',
    writeRelationshipType='SIMILAR_COSINE',
    writeProperty='score',
    relationshipWeightProperty="weight",
    sourceNodeFilter="Query",
    targetNodeFilter="Article",
)
st.header("node similarity (COSINE)")
st.write(f"Relationships produced: {result['relationshipsWritten']}")
st.write(f"Nodes compared: {result['nodesCompared']}")
st.write(f"Mean similarity: {result['similarityDistribution']['mean']}")

##############################
### evaluate (cosine similarity) ###
##############################

query = """
MATCH (q:Query)-[r:SIMILAR_COSINE]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""
st.header("evaluate (cosine similarity)")
st.write(gds.run_cypher(query))

##############################
### ppr (personalized pagerank) ###
##############################

st.header("ppr (personalized pagerank)")
for idx, name in enumerate(list(QUERY_DICT.keys())):
    nodeid = gds.find_node_id(labels=["Query"], properties={"name": name})
    result = gds.pageRank.write(
        G,
        writeProperty="pr"+str(idx),
        maxIterations=20,
        dampingFactor=0.85,
        relationshipWeightProperty='weight',
        sourceNodes=[nodeid]
    )   
    st.write(f"Node properties written: {result['nodePropertiesWritten']}")
    st.write(f"Mean: {result['centralityDistribution']['mean']}")

##############################
### evaluate (ppr) ###
##############################

query_idx = 0 
query_one = list(QUERY_DICT.keys())[query_idx]
query = f"""
MATCH (q:Query)-[r:CORRELATES]-(a:Article) WHERE q.name = "{query_one}"
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, a.pr{query_idx} AS ppr
ORDER BY Query, ppr DESC
LIMIT 10
"""
st.header("evaluate (ppr)")
st.write(gds.run_cypher(query))

##############################
### 1. node embedding ###
##############################

# fastrp
result = gds.fastRP.stream(
    G,
    randomSeed=42,
    embeddingDimension=16,
    relationshipWeightProperty="weight",
    iterationWeights=[1, 1, 1],
)

# node2vec
result = gds.node2vec.stream(
    G,
    randomSeed=42,
    embeddingDimension=16,
    relationshipWeightProperty="weight",
    iterations=3,
)

# hashgnn
result = gds.beta.hashgnn.stream(
    G,
    iterations = 3,
    embeddingDensity = 8,
    generateFeatures = {"dimension": 16, "densityLevel": 1},
    randomSeed = 42,
)

# st.write(f"Embedding vectors: {result['embedding']}")

# fastrp
result = gds.fastRP.mutate(
    G,
    mutateProperty="embedding_fastrp",
    randomSeed=42,
    embeddingDimension=16,
    relationshipWeightProperty="weight", # each relationship should have
    iterationWeights=[1, 1, 1],
)

# node2vec
result = gds.node2vec.mutate(
    G,
    mutateProperty="embedding_node2vec",
    randomSeed=42,
    embeddingDimension=16,
    relationshipWeightProperty="weight",
    iterations=3,
)

# hashgnn
result = gds.beta.hashgnn.mutate(
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

##############################
### 2. kNN ###
##############################

# fastrp
result = gds.knn.filtered.write(
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
result = gds.knn.filtered.write(
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
result = gds.knn.filtered.write(
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
st.write(gds.run_cypher(query))

# node2vec
query = """
MATCH (q:Query)-[r:SIMILAR_NODE2VEC]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""
st.header("evaluate (node2vec)")
st.write(gds.run_cypher(query))

# hashgnn
query = """
MATCH (q:Query)-[r:SIMILAR_HASHGNN]-(a:Article)
RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
ORDER BY Query, Similarity DESC
LIMIT 10
"""
st.header("evaluate (hashgnn)")
st.write(gds.run_cypher(query))


##############################
### free up memory ###
##############################

def free_up_memory():
    exists_result = gds.graph.exists(graph_name)
    if exists_result["exists"]:
        G = gds.graph.get(graph_name)
        G.drop()    
    query = """
    MATCH (n) DETACH DELETE n
    """
    gds.run_cypher(query)
    gds.close()
    st.sidebar.write("gds closed")
    st.session_state["reboot"] = True
st.sidebar.button("Free up memory", type="primary", on_click=free_up_memory) 

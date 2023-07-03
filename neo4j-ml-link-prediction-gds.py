from graphdatascience import GraphDataScience

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 

####################################
# # neo4j desktop (5.3.0, m1)
####################################

host = "bolt://localhost:7687"
user = "neo4j"
password= "j4oenj4oen"

gds = GraphDataScience(host, auth=(user, password))
print(gds.version())

####################################
# # 1. Import Data into Neo4j
####################################

query = """
CREATE CONSTRAINT constraint_article IF NOT EXISTS
For (article:Article) REQUIRE article.index IS UNIQUE
"""
gds.run_cypher(query)

query = """
CREATE CONSTRAINT constraint_author IF NOT EXISTS 
For (author:Author) REQUIRE author.name IS UNIQUE
"""
gds.run_cypher(query)

query = """
CALL apoc.periodic.iterate(
'UNWIND ["dblp-ref-0.json","dblp-ref-1.json", "dblp-ref-2.json","dblp-ref-3.json"] AS file 
CALL apoc.load.json("file:///" + file) 
YIELD value WHERE value.venue IN ["Lecture Notes in Computer Science", "Communications of The ACM", "international conference on software engineering", "advances in computing and communications"]
return value', 
'MERGE (a:Article {index:value.id}) ON CREATE SET a += apoc.map.clean(value,["id","authors","references"],[0]) WITH a, value.authors as authors 
UNWIND authors as author 
MERGE (b:Author {name:author}) 
MERGE (b)<-[:AUTHOR]-(a)', 
{batchSize:10000, parallel:true});
"""
gds.run_cypher(query)

####################################
# # 2. The Coauthorship Graph
####################################

query = """
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author) 
WITH a1, a2, paper 
ORDER BY a1, paper.year 
WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations 
MERGE (a1)-[coauthor:CO_AUTHOR {year: year}]-(a2) 
SET coauthor.collaborations = collaborations;
"""
gds.run_cypher(query)

####################################
# # 3. Create Training and Testing Datasets
####################################

query = """
MATCH (article:Article) 
RETURN article.year AS year, count(*) AS count
ORDER BY year 
"""
by_year = gds.run_cypher(query)

ax = by_year.plot(kind='bar', x='year', y='count', legend=None, figsize=(15,8)) 
ax.xaxis.set_label_text("") 
plt.tight_layout() 
# plt.show()
plt.savefig("figs/by_year.jpg")

query = """
MATCH (article:Article) 
RETURN article.year < 2006 AS training, count(*) AS count
"""
gds.run_cypher(query)

query = """
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author) 
WITH a1, a2, paper 
ORDER BY a1, paper.year 
WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations WHERE year < 2006
MERGE (a1)-[coauthor:CO_AUTHOR_EARLY {year: year}]-(a2) 
SET coauthor.collaborations = collaborations;
"""
gds.run_cypher(query)

query = """
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author) 
WITH a1, a2, paper 
ORDER BY a1, paper.year 
WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations WHERE year >= 2006
MERGE (a1)-[coauthor:CO_AUTHOR_LATE {year: year}]-(a2) 
SET coauthor.collaborations = collaborations;
"""
gds.run_cypher(query)

query = """
MATCH ()-[:CO_AUTHOR_EARLY]->() 
RETURN count(*) AS count
"""
gds.run_cypher(query)

query = """
MATCH ()-[:CO_AUTHOR_LATE]->() 
RETURN count(*) AS count
"""
gds.run_cypher(query)                                    

####################################
# # 4. Project Graph
####################################

exists_result = gds.graph.exists("myGraph")
if exists_result["exists"]:
    G = gds.graph.get("myGraph")
    G.drop()

G, _ = gds.graph.project(
    "myGraph",
    ["Author"],
    {"CO_AUTHOR_EARLY": {"orientation": "UNDIRECTED"}, "CO_AUTHOR": {"orientation": "UNDIRECTED"}}
)

gds.triangleCount.write(
    G,
    nodeLabels=["Author"],
    relationshipTypes=["CO_AUTHOR_EARLY"],
    writeProperty="trianglesTrain"
)

gds.triangleCount.write(
    G,
    nodeLabels=["Author"],
    relationshipTypes=["CO_AUTHOR"],
    writeProperty="trianglesTest"
)

gds.localClusteringCoefficient.write(
    G,
    nodeLabels=["Author"],
    relationshipTypes=["CO_AUTHOR_EARLY"],
    writeProperty="coefficientTrain"
)

gds.localClusteringCoefficient.write(
    G,
    nodeLabels=["Author"],
    relationshipTypes=["CO_AUTHOR"],
    writeProperty="coefficientTest"
)

gds.labelPropagation.write(
    G,
    nodeLabels=["Author"],
    relationshipTypes=["CO_AUTHOR_EARLY"],
    writeProperty="partitionTrain"
)

gds.labelPropagation.write(
    G,
    nodeLabels=["Author"],
    relationshipTypes=["CO_AUTHOR"],
    writeProperty="partitionTest"
)

query = """
CALL gds.louvain.stream("myGraph", {
  nodeLabels: ["Author"],
  relationshipTypes: ["CO_AUTHOR_EARLY"],
  includeIntermediateCommunities: true
})
YIELD nodeId, intermediateCommunityIds
WITH gds.util.asNode(nodeId) AS node, intermediateCommunityIds[0] AS smallestCommunity
SET node.louvainTrain = smallestCommunity
"""
gds.run_cypher(query)

query = """
CALL gds.louvain.stream("myGraph", {
  nodeLabels: ["Author"],
  relationshipTypes: ["CO_AUTHOR"],
  includeIntermediateCommunities: true
})
YIELD nodeId, intermediateCommunityIds
WITH gds.util.asNode(nodeId) AS node, intermediateCommunityIds[0] AS smallestCommunity
SET node.louvainTest = smallestCommunity
"""
gds.run_cypher(query)

G.drop()
G, _ = gds.graph.project(
    "myGraph",
    {"Author": {"properties": ["trianglesTrain", "trianglesTest", "coefficientTrain", "coefficientTest", "partitionTrain", "partitionTest", "louvainTrain", "louvainTest"]}},
    {"CO_AUTHOR_EARLY": {"orientation": "UNDIRECTED"}, "CO_AUTHOR": {"orientation": "UNDIRECTED"}},
)

####################################
# # 5. Create a Machine Learning Pipeline
####################################

exists_result = gds.beta.pipeline.exists("lp_pipe_fastrp")
if exists_result["exists"]:
    lp_pipe_fastrp = gds.pipeline.get("lp_pipe_fastrp")
    lp_pipe_fastrp.drop()

lp_pipe_fastrp = gds.lp_pipe("lp_pipe_fastrp")
lp_pipe_fastrp.addNodeProperty(
    "beta.hashgnn",
    mutateProperty="embedding",
    featureProperties=["trianglesTrain", "trianglesTest", "coefficientTrain", "coefficientTest", "partitionTrain", "partitionTest", "louvainTrain", "louvainTest"],
    heterogeneous=True,
    iterations=4,
    embeddingDensity=8,
    binarizeFeatures={"dimension": 8, "threshold": 0},
    randomSeed=42,
)
lp_pipe_fastrp.addFeature("hadamard", nodeProperties=["embedding"])
lp_pipe_fastrp.configureSplit(testFraction=0.2, validationFolds=5)
lp_pipe_fastrp.addRandomForest(numberOfDecisionTrees=30, maxDepth=10)

####################################
# # 6. Train
####################################

lp_model_fastrp, lp_stats_fastrp = lp_pipe_fastrp.train(
    G,
    modelName="lp_model_fastrp",
    targetRelationshipType="CO_AUTHOR_EARLY",
    randomSeed=42,
)

####################################
# # 7. Predict Links
####################################

metrics = lp_model_fastrp.metrics()
assert "AUCPR" in metrics
mutate_result = lp_model_fastrp.predict_mutate(G, topN=5, mutateRelationshipType="PRED_REL")
assert mutate_result["relationshipsWritten"] == 5 * 2  # Undirected relationships

####################################
# # (postprocessing) free up memory
####################################

lp_pipe_fastrp.drop()
lp_model_fastrp.drop()
G.drop()
query = """
MATCH (n) DETACH DELETE n
"""
gds.run_cypher(query)
gds.close()

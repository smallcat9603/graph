import py2neo as pg
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import igraph as ig
from numpy.random import randint

from pyspark.shell import spark
from pyspark.ml import Pipeline 
from pyspark.ml.classification import RandomForestClassifier 
from pyspark.ml.feature import StringIndexer, VectorAssembler 
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import * 
from pyspark.sql import functions as F
from sklearn.metrics import roc_curve, auc 
from collections import Counter
from cycler import cycler 

import matplotlib.pyplot as plt

# neo4j linux
# graph = pg.Graph("bolt://localhost:7687", auth=("neo4j", "j4oen")) # defaultly connect to database "neo4j", modify it in conf/neo4j.conf
# neo4j desktop m1
graph = pg.Graph("bolt://localhost:7687", auth=("neo4j", "j4oenj4oen"))

###
# 1. Importing the Data into Neo4j
###

query = """
CREATE CONSTRAINT constraint_article IF NOT EXISTS
For (article:Article) REQUIRE article.index IS UNIQUE
"""
graph.run(query)

query = """
CREATE CONSTRAINT constraint_author IF NOT EXISTS 
For (author:Author) REQUIRE author.name IS UNIQUE
"""
graph.run(query)

query = """
CALL apoc.periodic.iterate(
'UNWIND ["dblp-ref-0.json","dblp-ref-1.json", "dblp-ref-2.json","dblp-ref-3.json"] AS file 
CALL apoc.load.json("file:///" + file) 
YIELD value WHERE value.venue IN ["Lecture Notes in Computer Science", "Communications of The ACM", "international conference on software engineering", "advances in computing and communications"]
return value', 
'MERGE (a:Article {index:value.id}) ON CREATE SET a += apoc.map.clean(value,["id","authors","references"],[0]) WITH a,value.authors as authors 
UNWIND authors as author 
MERGE (b:Author{name:author}) 
MERGE (b)<-[:AUTHOR]-(a)', 
{batchSize: 10000, iterateList: true});
"""
graph.run(query)

###
# 2. The Coauthorship Graph
###

query = """
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author) 
WITH a1, a2, paper 
ORDER BY a1, paper.year 
WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations 
MERGE (a1)-[coauthor:CO_AUTHOR {year: year}]-(a2) 
SET coauthor.collaborations = collaborations;
"""
graph.run(query)

###
# 3. Creating Balanced Training and Testing Datasets
###

query = """
MATCH (article:Article) 
RETURN article.year AS year, count(*) AS count
ORDER BY year 
"""
by_year = graph.run(query).to_data_frame()

plt.style.use('fivethirtyeight') 
ax = by_year.plot(kind='bar', x='year', y='count', legend=None, figsize=(15,8)) 
ax.xaxis.set_label_text("") 
plt.tight_layout() 
plt.savefig("/Users/smallcat/Documents/GitHub/graph/fig1.jpg")
# plt.show()

query = """
MATCH (article:Article) 
RETURN article.year < 2006 AS training, count(*) AS count
"""
graph.run(query)

query = """
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author) 
WITH a1, a2, paper 
ORDER BY a1, paper.year 
WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations WHERE year < 2006
MERGE (a1)-[coauthor:CO_AUTHOR_EARLY {year: year}]-(a2) 
SET coauthor.collaborations = collaborations;
"""
graph.run(query)

query = """
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author) 
WITH a1, a2, paper 
ORDER BY a1, paper.year 
WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations WHERE year >= 2006
MERGE (a1)-[coauthor:CO_AUTHOR_LATE {year: year}]-(a2) 
SET coauthor.collaborations = collaborations;
"""
graph.run(query)

query = """
MATCH ()-[:CO_AUTHOR_EARLY]->() 
RETURN count(*) AS count
"""
graph.run(query)

query = """
MATCH ()-[:CO_AUTHOR_LATE]->() 
RETURN count(*) AS count
"""
graph.run(query)

def down_sample(df): 
    copy = df.copy() 
    zero = Counter(copy.label.values)[0] 
    un = Counter(copy.label.values)[1] 
    n = zero - un 
    copy = copy.drop(copy[copy.label == 0].sample(n=n, random_state=1).index) 
    return copy.sample(frac=1)

train_existing_links = graph.run(""" 
MATCH (author:Author)-[:CO_AUTHOR_EARLY]->(other:Author)
RETURN id(author) AS node1, id(other) AS node2, 1 AS label 
""").to_data_frame()

train_missing_links = graph.run(""" 
MATCH (author:Author) WHERE (author)-[:CO_AUTHOR_EARLY]-() 
MATCH (author)-[:CO_AUTHOR_EARLY*2..3]-(other) WHERE not((author)-[:CO_AUTHOR_EARLY]-(other))
RETURN id(author) AS node1, id(other) AS node2, 0 AS label 
""").to_data_frame()

train_missing_links = train_missing_links.drop_duplicates() 
training_df = train_missing_links.append(train_existing_links, ignore_index=True) 
training_df['label'] = training_df['label'].astype('category') 
training_df = down_sample(training_df) 
training_data = spark.createDataFrame(training_df)

training_data.show(n=5)
training_data.groupby("label").count().show()

test_existing_links = graph.run("""
MATCH (author:Author)-[:CO_AUTHOR_LATE]->(other:Author)
RETURN id(author) AS node1, id(other) AS node2, 1 AS label 
""").to_data_frame()

test_missing_links = graph.run(""" 
MATCH (author:Author) WHERE (author)-[:CO_AUTHOR_LATE]-() 
MATCH (author)-[:CO_AUTHOR*2..3]-(other) WHERE not((author)-[:CO_AUTHOR]-(other))
RETURN id(author) AS node1, id(other) AS node2, 0 AS label 
""").to_data_frame()

test_missing_links = test_missing_links.drop_duplicates() 
test_df = test_missing_links.append(test_existing_links, ignore_index=True) 
test_df['label'] = test_df['label'].astype('category') 
test_df = down_sample(test_df) 
test_data = spark.createDataFrame(test_df)

test_data.groupby("label").count().show()

###
# 4. How We Predict Missing Links
# the most predictive features will be related to communities
###

###
# 5. Creating a Machine Learning Pipeline
###

def create_pipeline(fields): 
    assembler = VectorAssembler(inputCols=fields, outputCol="features") 
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=30, maxDepth=10)
    return Pipeline(stages=[assembler, rf])

###
# 6. Predicting Links: Basic Graph Features
###

def apply_graphy_training_features(data): 
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1 
    MATCH (p2) WHERE id(p2) = pair.node2 
    RETURN pair.node1 AS node1, 
           pair.node2 AS node2,
           size([(p1)-[:CO_AUTHOR_EARLY]-(a)-[:CO_AUTHOR_EARLY]-(p2) | a]) AS commonAuthors, 
           size((p1)-[:CO_AUTHOR_EARLY]-()) * size((p2)- [:CO_AUTHOR_EARLY]-()) AS prefAttachment,
           size(apoc.coll.toSet([(p1)-[:CO_AUTHOR_EARLY]-(a) | id(a)] + [(p2)-[:CO_AUTHOR_EARLY]-(a) | id(a)])) AS totalNeighbors 
    """
    pairs = [{"node1": row["node1"], "node2": row["node2"]} for row in data.collect()]
    features = spark.createDataFrame(graph.run(query, {"pairs": pairs}).to_data_frame())
    return data.join(features, ["node1", "node2"])

def apply_graphy_test_features(data): 
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1 
    MATCH (p2) WHERE id(p2) = pair.node2 
    RETURN pair.node1 AS node1, 
           pair.node2 AS node2,
           size([(p1)-[:CO_AUTHOR]-(a)-[:CO_AUTHOR]-(p2) | a]) AS commonAuthors, 
           size((p1)-[:CO_AUTHOR]-()) * size((p2)-[:CO_AUTHOR]-()) AS prefAttachment,
           size(apoc.coll.toSet([(p1)-[:CO_AUTHOR]-(a) | id(a)] + [(p2)-[:CO_AUTHOR]-(a) | id(a)] )) AS totalNeighbors
    """
    pairs = [{"node1": row["node1"], "node2": row["node2"]} for row in data.collect()]
    features = spark.createDataFrame(graph.run(query, {"pairs": pairs}).to_data_frame()) 
    return data.join(features, ["node1", "node2"])

training_data = apply_graphy_training_features(training_data) 
test_data = apply_graphy_test_features(test_data)

plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True) 
charts = [(1, "have collaborated"), (0, "haven't collaborated")]
for index, chart in enumerate(charts): 
    label, title = chart 
    filtered = training_data.filter(training_data["label"] == label) 
    common_authors = filtered.toPandas()["commonAuthors"] 
    histogram = common_authors.value_counts().sort_index() 
    histogram /= float(histogram.sum()) 
    histogram.plot(kind="bar", x='Common Authors', color="darkblue", ax=axs[index], title=f"Authors who {title} (label={label})")
    axs[index].xaxis.set_label_text("Common Authors")
plt.tight_layout() 
plt.savefig("/Users/smallcat/Documents/GitHub/graph/fig2.jpg")
# plt.show()

def train_model(fields, training_data): 
    pipeline = create_pipeline(fields) 
    model = pipeline.fit(training_data) 
    return model

basic_model = train_model(["commonAuthors"], training_data)
eval_df = spark.createDataFrame( [(0,), (1,), (2,), (10,), (100,)], ['commonAuthors'])
basic_model.transform(eval_df).select("commonAuthors", "probability", "prediction") .show(truncate=False)

def evaluate_model(model, test_data): 
    # Execute the model against the test set 
    predictions = model.transform(test_data)
    # Compute true positive, false positive, false negative counts 
    tp = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
    fp = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
    fn = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
    # Compute recall and precision manually 
    recall = float(tp) / (tp + fn) 
    precision = float(tp) / (tp + fp)
    # Compute accuracy using Spark MLLib's binary classification evaluator 
    accuracy = BinaryClassificationEvaluator().evaluate(predictions)
    # Compute false positive rate and true positive rate using sklearn functions 
    labels = [row["label"] for row in predictions.select("label").collect()] 
    preds = [row["probability"][1] for row in predictions.select ("probability").collect()]
    fpr, tpr, threshold = roc_curve(labels, preds) 
    roc_auc = auc(fpr, tpr)
    return { "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "accuracy": accuracy, "recall": recall, "precision": precision }

def display_results(results): 
    results = {k: v for k, v in results.items() if k not in ["fpr", "tpr", "roc_auc"]}
    return pd.DataFrame({"Measure": list(results.keys()), "Score": list(results.values())})

basic_results = evaluate_model(basic_model, test_data) 
display_results(basic_results)

def create_roc_plot(): 
    plt.style.use('classic') 
    fig = plt.figure(figsize=(13, 8)) 
    plt.xlim([0, 1]) 
    plt.ylim([0, 1]) 
    plt.ylabel('True Positive Rate') 
    plt.xlabel('False Positive Rate') 
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'c', 'm', 'y', 'k'])))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random score (AUC = 0.50)')
    return plt, fig

def add_curve(plt, title, fpr, tpr, roc): 
    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc:0.2})")

plt, fig = create_roc_plot()
add_curve(plt, "Common Authors", basic_results["fpr"], basic_results["tpr"], basic_results["roc_auc"])
plt.legend(loc='lower right') 
plt.savefig("/Users/smallcat/Documents/GitHub/graph/fig3.jpg")
# plt.show()

training_data.filter(training_data["label"]==1).describe().select("summary", "commonAuthors", "prefAttachment", "totalNeighbors").show()
training_data.filter(training_data["label"]==0).describe().select("summary", "commonAuthors", "prefAttachment", "totalNeighbors").show()

fields = ["commonAuthors", "prefAttachment", "totalNeighbors"] 
graphy_model = train_model(fields, training_data)
graphy_results = evaluate_model(graphy_model, test_data) 
display_results(graphy_results)

plt, fig = create_roc_plot()
add_curve(plt, "Common Authors", basic_results["fpr"], basic_results["tpr"], basic_results["roc_auc"])
add_curve(plt, "Graphy", graphy_results["fpr"], graphy_results["tpr"], graphy_results["roc_auc"])
plt.legend(loc='lower right') 
plt.savefig("/Users/smallcat/Documents/GitHub/graph/fig4.jpg")
# plt.show()

def plot_feature_importance(fields, feature_importances): 
    feature_importances_new = [feature_importances[i] for i in range(len(feature_importances))]
    df = pd.DataFrame({"Feature": fields, "Importance": feature_importances_new}) 
    df = df.sort_values("Importance", ascending=False) 
    ax = df.plot(kind='bar', x='Feature', y='Importance', legend=None) 
    ax.xaxis.set_label_text("") 
    plt.tight_layout() 
    plt.savefig("/Users/smallcat/Documents/GitHub/graph/fig5.jpg")
    # plt.show()

rf_model = graphy_model.stages[-1] 
plot_feature_importance(fields, rf_model.featureImportances)

###
# 7. Predicting Links: Triangles and the Clustering Coefficient
###

query = """
CALL gds.graph.drop("myGraph")
"""
graph.run(query)

query = """
CALL gds.graph.project(
  'myGraph',    
  ['Author'],   
  {CO_AUTHOR_EARLY: {orientation: 'UNDIRECTED'}, CO_AUTHOR: {orientation: 'UNDIRECTED'}}    
)
"""
graph.run(query)

query = """
CALL gds.triangleCount.write("myGraph", {
  nodeLabels: ["Author"],
  relationshipTypes: ["CO_AUTHOR_EARLY"],
  writeProperty: 'trianglesTrain'
})
"""
graph.run(query)

query = """
CALL gds.triangleCount.write("myGraph", {
  nodeLabels: ["Author"],
  relationshipTypes: ["CO_AUTHOR"],
  writeProperty: 'trianglesTest'
})
"""
graph.run(query)

query = """
CALL gds.localClusteringCoefficient.write("myGraph", {
  nodeLabels: ["Author"],
  relationshipTypes: ["CO_AUTHOR_EARLY"],
  writeProperty: 'coefficientTrain'
})
"""
graph.run(query)

query = """
CALL gds.localClusteringCoefficient.write("myGraph", {
  nodeLabels: ["Author"],
  relationshipTypes: ["CO_AUTHOR"],
  writeProperty: 'coefficientTest'
})
"""
graph.run(query)

def apply_triangles_features(data, triangles_prop, coefficient_prop): 
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1 
    MATCH (p2) WHERE id(p2) = pair.node2 
    RETURN pair.node1 AS node1, 
           pair.node2 AS node2,
           apoc.coll.min([p1[$trianglesProp], p2[$trianglesProp]]) AS minTriangles,
           apoc.coll.max([p1[$trianglesProp], p2[$trianglesProp]]) AS maxTriangles,
           apoc.coll.min([p1[$coefficientProp], p2[$coefficientProp]]) AS minCoefficient,
           apoc.coll.max([p1[$coefficientProp], p2[$coefficientProp]]) AS maxCoefficient
    """
    params = { "pairs": [{"node1": row["node1"], "node2": row["node2"]} for row in data.collect()],
               "trianglesProp": triangles_prop, 
               "coefficientProp": coefficient_prop}
    features = spark.createDataFrame(graph.run(query, params).to_data_frame()) 
    return data.join(features, ["node1", "node2"])

training_data = apply_triangles_features(training_data, "trianglesTrain", "coefficientTrain")
test_data = apply_triangles_features(test_data, "trianglesTest", "coefficientTest")

training_data.filter(training_data["label"]==1).describe().select("summary", "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient").show()
training_data.filter(training_data["label"]==0).describe().select("summary", "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient").show()

fields = ["commonAuthors", "prefAttachment", "totalNeighbors", "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient"]
triangle_model = train_model(fields, training_data)
triangle_results = evaluate_model(triangle_model, test_data) 
display_results(triangle_results)

plt, fig = create_roc_plot()
add_curve(plt, "Common Authors", basic_results["fpr"], basic_results["tpr"], basic_results["roc_auc"])
add_curve(plt, "Graphy", graphy_results["fpr"], graphy_results["tpr"], graphy_results["roc_auc"])
add_curve(plt, "Triangles", triangle_results["fpr"], triangle_results["tpr"], triangle_results["roc_auc"])
plt.legend(loc='lower right') 
plt.savefig("/Users/smallcat/Documents/GitHub/graph/fig6.jpg")
# plt.show()

rf_model = triangle_model.stages[-1] 
plot_feature_importance(fields, rf_model.featureImportances)

###
# 8. Predicting Links: Community Detection
###

query = """
CALL gds.labelPropagation.write("myGraph", {
  nodeLabels: ["Author"],
  relationshipTypes: ["CO_AUTHOR_EARLY"],
  writeProperty: 'partitionTrain'
})
"""
graph.run(query)

query = """
CALL gds.labelPropagation.write("myGraph", {
  nodeLabels: ["Author"],
  relationshipTypes: ["CO_AUTHOR"],
  writeProperty: 'partitionTest'
})
"""
graph.run(query)

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
graph.run(query)

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
graph.run(query)

def apply_community_features(data, partition_prop, louvain_prop): 
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1 
    MATCH (p2) WHERE id(p2) = pair.node2 
    RETURN pair.node1 AS node1, 
           pair.node2 AS node2,
           CASE WHEN p1[$partitionProp] = p2[$partitionProp] THEN 1 ELSE 0 END AS samePartition,
           CASE WHEN p1[$louvainProp] = p2[$louvainProp] THEN 1 ELSE 0 END AS sameLouvain
    """
    params = { "pairs": [{"node1": row["node1"], "node2": row["node2"]} for row in data.collect()],
               "partitionProp": partition_prop, 
               "louvainProp": louvain_prop
    }
    features = spark.createDataFrame(graph.run(query, params).to_data_frame()) 
    return data.join(features, ["node1", "node2"])

training_data = apply_community_features(training_data, "partitionTrain", "louvainTrain")
test_data = apply_community_features(test_data, "partitionTest", "louvainTest")

plt.style.use('fivethirtyeight') 
fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True) 
charts = [(1, "have collaborated"), (0, "haven't collaborated")]

for index, chart in enumerate(charts): 
    label, title = chart
    filtered = training_data.filter(training_data["label"] == label) 
    values = (filtered.withColumn('samePartition', F.when(F.col("samePartition") == 0, "False").otherwise("True")).groupby("samePartition").agg(F.count("label").alias("count")).select("samePartition", "count").toPandas())
    values.set_index("samePartition", drop=True, inplace=True) 
    values.plot(kind="bar", ax=axs[index], legend=None, title=f"Authors who {title} (label={label})")
    axs[index].xaxis.set_label_text("Same Partition")

plt.tight_layout() 
plt.savefig("/Users/smallcat/Documents/GitHub/graph/fig7.jpg")
# plt.show()

for index, chart in enumerate(charts): 
    label, title = chart 
    filtered = training_data.filter(training_data["label"] == label) 
    values = (filtered.withColumn('sameLouvain', F.when(F.col("sameLouvain") == 0, "False").otherwise("True")).groupby("sameLouvain").agg(F.count("label").alias("count")).select("sameLouvain", "count") .toPandas())
    values.set_index("sameLouvain", drop=True, inplace=True) 
    values.plot(kind="bar", ax=axs[index], legend=None, title=f"Authors who {title} (label={label})")
    axs[index].xaxis.set_label_text("Same Louvain")

plt.tight_layout() 
plt.savefig("/Users/smallcat/Documents/GitHub/graph/fig8.jpg")
# plt.show()

fields = ["commonAuthors", "prefAttachment", "totalNeighbors", "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient", "samePartition", "sameLouvain"]
community_model = train_model(fields, training_data)

community_results = evaluate_model(community_model, test_data) 
display_results(community_results)

plt, fig = create_roc_plot()
add_curve(plt, "Common Authors", basic_results["fpr"], basic_results["tpr"], basic_results["roc_auc"])
add_curve(plt, "Graphy", graphy_results["fpr"], graphy_results["tpr"], graphy_results["roc_auc"])
add_curve(plt, "Triangles", triangle_results["fpr"], triangle_results["tpr"], triangle_results["roc_auc"])
add_curve(plt, "Community", community_results["fpr"], community_results["tpr"], community_results["roc_auc"])
plt.legend(loc='lower right') 
plt.savefig("/Users/smallcat/Documents/GitHub/graph/fig9.jpg")
# plt.show()

rf_model = community_model.stages[-1] 
plot_feature_importance(fields, rf_model.featureImportances)
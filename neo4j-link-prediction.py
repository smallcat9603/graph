import py2neo as pg
import pandas as pd
import igraph as ig

from numpy.random import randint
from pyspark.ml import Pipeline 
from pyspark.ml.classification import RandomForestClassifier 
from pyspark.ml.feature import StringIndexer, VectorAssembler 
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import * 
from pyspark.sql import functions as F
from sklearn.metrics import roc_curve, auc 
from collections import Counter
from cycler import cycler 
# import matplotlib 
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from pyspark.shell import spark

# neo4j linux
# graph = pg.Graph("bolt://localhost:7687", auth=("neo4j", "j4oen")) # defaultly connect to database "neo4j", modify it in conf/neo4j.conf
# neo4j desktop m1
graph = pg.Graph("bolt://localhost:7687", auth=("neo4j", "j4oenj4oen"))

###
# 1. Importing the Data into Neo4j
###

query = """
CREATE CONSTRAINT ON (article:Article) 
ASSERT article.index IS UNIQUE;
"""
graph.run(query)

query = """
CREATE CONSTRAINT ON (author:Author) 
ASSERT author.name IS UNIQUE;
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
plt.savefig("test.jpg")
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
plt.savefig("hist.jpg")
# plt.show()

def train_model(fields, training_data): 
    pipeline = create_pipeline(fields) 
    model = pipeline.fit(training_data) 
    return model

basic_model = train_model(["commonAuthors"], training_data)
eval_df = spark.createDataFrame( [(0,), (1,), (2,), (10,), (100,)], ['commonAuthors'])
basic_model.transform(eval_df).select("commonAuthors", "probability", "prediction") .show(truncate=False)
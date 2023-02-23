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
# import matplotlib.pyplot as plt

graph = pg.Graph("bolt://localhost:7687", auth=("neo4j", "j4oen")) # defaultly connect to database "neo4j", modify it in conf/neo4j.conf

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
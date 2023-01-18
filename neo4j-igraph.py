import py2neo as pg
import igraph as ig

neo4j = pg.Graph("bolt://54.237.250.148:7687", auth=("neo4j", "fifths-effectiveness-shafts"))

query = """
LOAD CSV WITH HEADERS 
FROM 'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/HP/hp_1.csv' 
AS row
MERGE (s:Character {name:row.source})
MERGE (t:Character {name:row.target})
MERGE (s)-[i:INTERACTS]->(t)
SET i.weight = toInteger(row.weight)
"""
neo4j.run(query)

query = """
MATCH (c:Character)
RETURN c.name AS character,
       size((c)--()) AS degree
"""
data = neo4j.run(query).data() # dictlist
print(data)

query = """
MATCH (s)-[i:INTERACTS]->(t) WHERE i.weight > 5
RETURN s.name, t.name
"""
data = neo4j.run(query).to_data_frame()
print(data)

# neo4j -> igraph
g = ig.Graph.DataFrame(data, directed=True, use_vids=False)
print(g)
best = g.vs.select(_degree = g.maxdegree())["name"]
print(best)
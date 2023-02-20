import py2neo as pg
import igraph as ig

# local
# neo4j = pg.Graph("bolt://localhost:7687", auth=("neo4j", "j4oen"))
# neo4j sandbox
# neo4j = pg.Graph("bolt://3.237.242.164:7687", auth=("neo4j", "nomenclature-adhesive-removal"))
# neo4j aura
neo4j = pg.Graph("neo4j+s://3f3388c5.databases.neo4j.io", auth=("neo4j", "Zmf0G3eCjEWyxq6dbK6Q9h9PlHVJj_iFpojKkC0Mrps"))

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

# in the latest version of neo4j, size() should get a pattern comprehension as parameter, wrong: (c)--(), correct: [p=(c)--() | p]
query = """
MATCH (c:Character)
RETURN c.name AS character,
       size([p=(c)--() | p]) AS degree
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
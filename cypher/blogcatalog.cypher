CREATE CONSTRAINT id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.name IS UNIQUE;

LOAD CSV FROM "https://raw.githubusercontent.com/smallcat9603/graph/main/data/blogcatalog.edges" AS row FIELDTERMINATOR " "

MERGE (s:Node {name: row[0]})
MERGE (t:Node {name: row[1]})
MERGE (s)-[:EDGE]-(t)

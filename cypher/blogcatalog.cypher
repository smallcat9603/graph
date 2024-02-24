CREATE INDEX node_name FOR (n:Node) ON (n.name);

CALL apoc.periodic.iterate(
    "LOAD CSV FROM 'https://raw.githubusercontent.com/smallcat9603/graph/main/data/blogcatalog.edges' AS row FIELDTERMINATOR ' '
    RETURN row",
    "MERGE (s:Node {name: row[0]})
    MERGE (t:Node {name: row[1]})
    CREATE (s)-[:EDGE]-(t)",
    {batchSize: 1000}
)

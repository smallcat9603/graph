CREATE INDEX node_name FOR (n:Node) ON (n.name);

CALL apoc.periodic.iterate(
    "LOAD CSV FROM 'https://raw.githubusercontent.com/smallcat9603/graph/main/data/blogcatalog_0.edges' AS row FIELDTERMINATOR ' '
    RETURN row",
    "MERGE (s:Node {name: row[0]})
    MERGE (t:Node {name: row[1]})
    MERGE (s)-[:EDGE]-(t)",
    {batchSize: 1000}
);

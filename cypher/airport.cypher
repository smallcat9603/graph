CREATE INDEX node_name FOR (n:Node) ON (n.name);

CALL apoc.periodic.iterate(
    "LOAD CSV FROM 'https://raw.githubusercontent.com/smallcat9603/graph/main/data/airport_0.edges' AS row FIELDTERMINATOR ' '
    RETURN row",
    "MERGE (s:Node {name: row[0]})
    MERGE (t:Node {name: row[1]})
    MERGE (s)-[e:EDGE]-(t)
    SET e.weight = toInteger(row[2])",
    {batchSize: 1000}
);

CALL apoc.periodic.iterate(
    "LOAD CSV FROM 'https://raw.githubusercontent.com/smallcat9603/graph/main/data/airport.country_0.labels' AS row FIELDTERMINATOR ' '
    RETURN row",
    "MATCH (n:Node {name: row[0]})
    SET n.labels = coalesce(n.labels, []) + row[1]",
    {batchSize: 1000}    
);

CALL apoc.periodic.iterate(
    "LOAD CSV FROM 'https://raw.githubusercontent.com/smallcat9603/graph/main/data/airport.continent_0.labels' AS row FIELDTERMINATOR ' '
    RETURN row",
    "MATCH (n:Node {name: row[0]})
    SET n.labels = coalesce(n.labels, []) + row[1]",
    {batchSize: 1000}    
);

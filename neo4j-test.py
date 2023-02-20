from graphdatascience import GraphDataScience

import seaborn as sns
import matplotlib.pyplot as plt

host = "bolt://localhost:7687"
user = "neo4j"
password= "j4oen"

# https://sandbox.neo4j.com/?usecase=blank-sandbox&_ga=2.99281694.1083263989.1673158924-1227688473.1673158924&_gl=1*ln3csj*_ga*MjA0Nzc1MDI3My4xNjcyODkwMTI1*_ga_DL38Q8KGQC*MTY3MzE2NjEyMS4zLjEuMTY3MzE2OTM0My4wLjAuMA..
host = "bolt://54.226.99.132:7687"
user = "neo4j"
password= "traffic-state-regret"

gds = GraphDataScience(host, auth=(user, password))
print(gds.version())

query = """
call dbms.procedures()
"""
print(gds.run_cypher(query))

query = """
LOAD CSV WITH HEADERS FROM $url AS row
MERGE (s:Character {name:row.source})
MERGE (t:Character {name:row.target})
MERGE (s)-[i:INTERACTS]->(t)
SET i.weight = toInteger(row.weight)
"""
params = {'url': 'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/HP/hp_1.csv'}
gds.run_cypher(query, params)

degree_df = gds.run_cypher("""
MATCH (c:Character)
RETURN c.name AS character,
       size((c)--()) AS degree
""")

print(degree_df.head())

# sns.displot(data=degree_df, x="degree", height=7, aspect=1.5)

G, metadata = gds.graph.project( # created only once
    "hp-graph", 
    "Character",
    {"INTERACTS": {"orientation": "UNDIRECTED", "properties": ["weight"]}}
)

print(G.name()) # hp-graph
print(G.memory_usage()) # 2341 KiB
print(G.density()) # 0.05782652043868395
print(metadata)

pagerank_df = gds.pageRank.stream(G, relationshipWeightProperty="weight")
print(pagerank_df.head())

pagerank_df['node_object'] = gds.util.asNodes(pagerank_df['nodeId'].to_list())
# Extract name properties from the node object
pagerank_df['name'] = [n['name'] for n in pagerank_df['node_object']]
# Draw a bar chart
plt.figure(figsize=(16,9))
sns.barplot(x='name', y='score', data=pagerank_df.sort_values(by='score', ascending=False).head(10))
plt.xticks(rotation=45)

louvain_metadata = gds.louvain.mutate(G, mutateProperty='communityId', relationshipWeightProperty='weight')
print(louvain_metadata)
print(G.node_properties('Character')) # ['communityId']

louvain_df = gds.graph.streamNodeProperty(G, 'communityId')
print(louvain_df.head())

# Rename columns
louvain_df.columns = ['nodeId', 'communityId']
# You can do all sorts of pandas operations like aggregations
louvain_df.groupby("communityId").size().to_frame(
    "communitySize"
).reset_index().sort_values(by=["communitySize"], ascending=False)

print(gds.graph.list())
G = gds.graph.get("hp-graph")
# Drop a projected in-memory graph
G.drop()
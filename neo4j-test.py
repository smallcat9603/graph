from graphdatascience import GraphDataScience

# host = "bolt://54.226.180.149:7687"
# user = "neo4j"
# password= "arc-anticipation-yaws"

host = "bolt://localhost:7687"
user = "neo4j"
password= "j4oen"

gds = GraphDataScience(host, auth=(user, password))

print(gds.version())
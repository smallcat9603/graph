import Pyro4
import sys
import igraph as ig
import pandas as pd
import rw

# map to node ids that are from 0 and continuous
def map_nodes_in_edgelist(file, file_new):
    node_map = {}
    with open(file, "r") as f:
        edgelist = [line.strip().split() for line in f]
        edgelist = [list(map(int, edge)) for edge in edgelist]
        nodes = set([edge[0] for edge in edgelist] + [edge[1] for edge in edgelist])
        node_map = {node: i for i, node in enumerate(nodes)}
        edgelist_new = [(node_map[edge[0]], node_map[edge[1]]) for edge in edgelist]
        df = pd.DataFrame(edgelist_new)
        df.to_csv(file_new, sep=" ", index=False, header=False)
        return node_map

nservers = 3 # server id = 0,1,2
this = sys.argv[1]

if len(sys.argv) != 2 or not this.isdigit() or int(this) > nservers-1 or int(this) < 0:
    print("input server id [0-{0}]".format(nservers-1))
    sys.exit(1)

# standardize node id in edgelist file and read subgraph, otherwise graph is not connected in igraph
file = f"test.sub{this}.txt"
file_new = file.rstrip(".txt") + ".x.txt"
node_map = map_nodes_in_edgelist(file, file_new)
graph = ig.Graph.Read_Edgelist(file_new, directed=False)

# check graph
if(graph.is_connected()):
    print("Graph is connected")
    print(graph.summary())
else:
    print("Graph is not connected")
    sys.exit(1)

# read route table from file
columns = ["sources", "targets_servers"]
routes = pd.read_csv(f"test.rt{this}.txt", comment="#", sep="\s+", names=columns)
route_table = {}
for row in range(len(routes)):
    route_table[int(routes["sources"][row])] = list(eval(routes["targets_servers"][row]))

servername = "Server" + this
daemon = Pyro4.Daemon()
nhops = 100
obj = rw.Walker(this, graph, route_table, node_map, nhops)
uri = daemon.register(obj)
ns = Pyro4.locateNS()
ns.register(servername, uri)
# enter the service loop.
print("Server%s started ..." % this)
daemon.requestLoop()
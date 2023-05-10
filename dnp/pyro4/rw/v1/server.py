import Pyro5
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
        node_map = {node: i for i, node in enumerate(nodes)} # global --> local
        edgelist_new = [(node_map[edge[0]], node_map[edge[1]]) for edge in edgelist]
        df = pd.DataFrame(edgelist_new)
        df.to_csv(file_new, sep=" ", index=False, header=False)
        return node_map
    
# read hosts from file
columns = ["server_id", "ip_port"]
hostfile = pd.read_csv("hosts.txt", comment="#", sep="\s+", names=columns)
nhosts = len(hostfile) # server id = 0,1,2
hosts = {}
for row in range(nhosts):
    hosts[int(hostfile["server_id"][row])] = hostfile["ip_port"][row]

this = ""
if len(sys.argv) == 2 and 0 <= int(sys.argv[1]) < nhosts:
    this = sys.argv[1]
else:
    print("input server id [0-{0}]".format(nhosts-1))
    sys.exit(1)

# standardize node id in edgelist file and read subgraph, otherwise graph is not connected in igraph (global --> local)
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

# boot server
# servername = "Server" + this # this = 0, 1, 2, ...
# serverport = 9091 + serverid # ns port is 9090, server port is from 9091, server0 --> port9091, server1 --> port9092, server2 --> port9093, ...
serverid = int(this)
host_port = hosts[serverid].split(":")
daemon = Pyro5.server.Daemon(host=host_port[0], port=int(host_port[1]))
obj = rw.Walker(this, graph, route_table, node_map, hosts)
uri = daemon.register(obj, objectId="walker") # default objectId is random like obj_79549b4c52dc43ffaaa486b76b25c2af
# ns = Pyro5.core.locate_ns()
# ns.register(servername, uri)
# enter the service loop.
print("Server%s started ..." % this)
daemon.requestLoop()
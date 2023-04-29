import Pyro4
import random
import igraph as ig
import sys

class Walker(object):
    def __init__(self, name, graph, route_table, node_map, nhops): 
        self.name = name
        self.graph = graph
        self.route_table = route_table
        self.node_map = node_map # node_map = {100: 0, 200: 1, 150: 2, ...}, global --> local
        self.map_node = {v: k for k, v in node_map.items()} # reverse keys and values in node_map, map_node = {0: 100, 1: 200, 2: 150, ...}, local --> global
        self.nhops = nhops
        self.next = None
    def nexthop(self, cur_local, cur_global):
        neighbors_in = self.graph.neighbors(cur_local)
        nneighbors_in = len(neighbors_in)
        neighbors_out = self.route_table[cur_global]
        nneighbors_out = len(neighbors_out)
        nneighbors = nneighbors_in + nneighbors_out
        next_idx = random.randint(0, nneighbors-1)
        next_local_node = -1
        next_global_node = -1
        next_global_server = -1
        if next_idx < nneighbors_in: # next node is inside server
            next_local_node = neighbors_in[next_idx]
            next_global_node = self.map_node[next_local_node]
        else: # next node is outside server
            next_global = neighbors_out[next_idx-nneighbors_in]
            next_global_node = next_global[0]
            next_global_server = next_global[1]   
        return next_local_node, next_global_node, next_global_server
    @Pyro4.expose
    def walk(self, message): 
        next_local_node = -1
        next_global_node = -1
        next_global_server = -1
        while next_global_server == -1 and len(message) < self.nhops: # walk inside
            msg = message[-1]
            if msg == "go": # starting point of walker
                cur_local = random.randint(0, self.graph.vcount()-1)
                cur_global = self.map_node[cur_local]
                message = [cur_global]   
            elif isinstance(msg, int) and msg >= 0:
                cur_global = msg
                cur_local = self.node_map[cur_global]
            else:
                print("messge is wrong")
                sys.exit(1)
            next_local_node, next_global_node, next_global_server = self.nexthop(cur_local, cur_global)
            message.append(next_global_node)                       
        hops = len(message)
        if hops >= self.nhops:
            print("Walker stopped at Server{0}".format(self.name))
            print("Finished. Walker walks through nodes: {0}".format(message))
        elif next_local_node == -1: # walk outside
            print("Walker walks through {0} nodes".format(hops))
            nextname = str(next_global_server)
            print("Walker walks from Server{0} to Server{1}".format(self.name, nextname))
            self.next = Pyro4.Proxy("PYRONAME:Server" + nextname)
            self.next.walk(message)
        else:
            print("Something is wrong")
            sys.exit(1)

import Pyro5.server
import Pyro5.client
import random
import sys, os
import time

class Walker(object):
    def __init__(self, name, graph, route_table, node_map, hosts): # nodes in route_table are global
        self.name = name
        self.graph = graph
        self.route_table = route_table
        self.node_map = node_map # node_map = {100: 0, 200: 1, 150: 2, ...}, global --> local
        self.map_node = {v: k for k, v in node_map.items()} # reverse keys and values in node_map, map_node = {0: 100, 1: 200, 2: 150, ...}, local --> global
        self.go_out = 0
        self.hosts = hosts
        self.start_time = 0.0
        self.stop_time = 0.0
        self.timestamp = 0

    def nexthop_roulette(self, cur_local, cur_global):
        neighbors_in = self.graph.neighbors(cur_local)
        nneighbors_in = len(neighbors_in)
        nneighbors_out = 0
        if cur_global in self.route_table:
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
    
    def save_log(self, *items):
        dir = "../log"
        if not os.path.exists(dir):
            os.makedirs(dir)
        filepath = dir + f"/{self.timestamp}.log"
        line = ""
        for i in items:
            line += str(i) + "\t"
        line += "\n"
        with open(filepath, "a") as f:
            f.write(line)
        print(f"saved in {filepath}.")

    def save_path(self, walker, message):
        filepath = f"../log/{self.timestamp}.txt"
        with open(filepath, "a") as f:
            f.write(f'{walker}\t{message}\n')
        print(f"saved in {filepath}.")

    @Pyro5.server.expose
    @Pyro5.server.oneway
    def walk(self, message, nhops, walker): 
        next_local_node = -1
        next_global_node = -1
        next_global_server = -1
        while next_global_server == -1 and len(message) < nhops: # walk inside
            msg = message[-1]
            if isinstance(msg, str) and msg.startswith("go_"): # starting point of walker
                print(f"Walker{walker} gets started to walk at Server{self.name}")
                self.timestamp = int(msg[len("go_"):])
                self.start_time = time.time()
                self.go_out = 0
                cur_local = random.randint(0, self.graph.vcount()-1)
                cur_global = self.map_node[cur_local]
                message = [cur_global]   
            elif isinstance(msg, int) and msg >= 0:
                cur_global = msg
                cur_local = self.node_map[cur_global]
            else:
                print("messge is wrong")
                sys.exit(1)
            next_local_node, next_global_node, next_global_server = self.nexthop_roulette(cur_local, cur_global)
            message.append(next_global_node)                      
        if len(message) >= nhops:
            self.stop_time = time.time()
            print(f"Finished. Walker{walker} stopped at Server{self.name}, walking through {len(message)} nodes:\n{message}")
            self.save_log(self.timestamp,
                          self.graph.vcount(),
                          self.graph.ecount(),                          
                          self.start_time, 
                          self.stop_time,
                          self.stop_time-self.start_time,
                          len(self.hosts),
                          self.name,
                          self.go_out,
                          walker,
                          nhops)
            self.save_path(walker, message)
        elif next_local_node == -1: # walk outside
            nextname = str(next_global_server)
            self.go_out += 1
            # print(f"{self.go_out}: Walker{walker} walked through {len(message)} nodes, and will go from Server{self.name} to Server{nextname}")
            uri = "PYRO:walker@" + self.hosts[next_global_server]
            # with Pyro5.client.Proxy("PYRONAME:Server" + nextname) as next: # require ns
            with Pyro5.client.Proxy(uri) as next: # not require ns
                next.walk(message, nhops, walker)
        else:
            print(f"Something is wrong for Walker{walker}")
            sys.exit(1)

import Pyro5.server
import Pyro5.client
import random
import sys
import time
import concurrent.futures

# @Pyro5.server.behavior(instance_mode="single")
class Walker(object):
    def __init__(self, name, graph, max_threads, chunk_size, route_table, node_map, hosts): # nodes in route_table are global
        self.name = name
        self.graph = graph
        self.max_threads = max_threads
        self.chunk_size = chunk_size
        self.route_table = route_table
        self.node_map = node_map # node_map = {100: 0, 200: 1, 150: 2, ...}, global --> local
        self.map_node = {v: k for k, v in node_map.items()} # reverse keys and values in node_map, map_node = {0: 100, 1: 200, 2: 150, ...}, local --> global
        self.hosts = hosts
        self.nhosts = len(hosts)
        self.start_time = 0.0
        self.stop_time = 0.0
        self.paths = []
        self.go_out = 0

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

    def remote_invoke(self, next_global_server, message, nhops, walker):
        # nextname = str(next_global_server)
        # print(f"Walker{walker} walked through {len(message)} nodes, and will go from Server{self.name} to Server{nextname}")
        uri = "PYRO:walker@" + self.hosts[next_global_server]
        # with Pyro5.client.Proxy("PYRONAME:Server" + nextname) as next: # require ns
        with Pyro5.client.Proxy(uri) as next: # not require ns
            next.walk(message, nhops, walker)
            # try:
            #     next.walk(message, nhops, walker)
            # except Pyro5.errors.ConnectionClosedError:  # or possibly CommunicationError
            #     print(f"Connection lost. REBINDING to Server{next_global_server} ({self.hosts[next_global_server]})...")
            #     next._pyroReconnect()

    @Pyro5.server.expose
    def get_results(self):
        return self.start_time, self.stop_time, self.go_out, self.paths

    @Pyro5.server.expose
    # @Pyro5.server.oneway
    def walk(self, message, nhops, walker): 
        next_local_node = -1
        next_global_node = -1
        next_global_server = -1
        while next_global_server == -1 and len(message) < nhops: # walk inside
            msg = message[-1]
            if isinstance(msg, str) and msg == "go": # starting point of walker
                print(f"Walker{walker} gets started to walk at Server{self.name}")
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
            print(f"Finished. Walker{walker} stopped at Server{self.name}, walking through {len(message)} nodes.")
            self.stop_time = time.time()
            self.paths.append((walker, message))
        elif next_local_node == -1: # walk outside
            self.go_out += 1
            self.remote_invoke(next_global_server, message, nhops, walker)
        else:
            print(f"Something is wrong for Walker{walker}")
            sys.exit(1)

    @Pyro5.server.expose
    # @Pyro5.server.oneway
    def start_walkers(self, nhops, id_start, id_end): 
        self.start_time = time.time()
        time.sleep(0.001) # prevent arriving before starting
        print(f"Walkers[{id_start}-{id_end-1}] start at Server{self.name} ...")

        # # option1: stupid max_thread_num control by delay
        # for walker in range(id_start, id_end):
        #     t = threading.Thread(target=self.walk, args=[["go"], nhops, walker])
        #     t.daemon = True
        #     t.start()
        #     time.sleep(0.3) # avoid Pyro5.errors.ConnectionClosedError (m1 experience: 1-server --> 0.05, 2-server --> 0.1, 4-server --> 0.3)

        # option2: automatic max_thread_num control by ThreadPool submit
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
        #     for walker in range(id_start, id_end):
        #         executor.submit(self.walk, ["go"], nhops, walker)
        #     # executor.shutdown()

        # option3: automatic max_thread_num control by ThreadPool map
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            params = map(lambda x: (["go"], nhops, x), range(id_start, id_end))
            executor.map(lambda args: self.walk(*args), params, chunksize=self.chunk_size)
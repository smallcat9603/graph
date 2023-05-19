# smallcat 230515

import igraph as ig
import sys, getopt, os
import pandas as pd

def printUsage():
    print('Usage: python3 {0} <edgelistfile> <metisoutput>'.format(os.path.basename(__file__)))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h") # opts = [("-h", " ")], args = [edgelistfile, metisoutput]
    except getopt.GetoptError:
        printUsage()
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        else:
            printUsage()
            sys.exit(1)
    if len(args) != 2:
            printUsage()
            sys.exit(1)       
    edgefile = args[0]
    metisoutput = args[1]
    nsubs = metisoutput.split(".part.")[1]
    nsubgraphs = int(nsubs)

    # create graph based on edge file
    # DO NOT USE igraph._igraph.GraphBase, USE SUBCLASS igraph.Graph instead
    # Read_Edgelist() in igraph is 0-based !!! (number vertices from 0)
    G = ig.Graph.Read_Edgelist(edgefile, directed=False)

    # check graph
    if(G.is_connected()):
        print("Graph is connected")
    else:
        print("Graph is not connected")
        nnodes = G.vcount()
        G = G.components().giant()
        print("The largest component is used instead: " + str(G.vcount()/nnodes))
        sys.exit(1)

    # print graph info
    print(G.summary())

    # node "name" = id
    for v in G.vs:
        v["name"] = v.index

    with open(metisoutput, 'r') as file:
        lines = file.readlines()
    membership = [int(line.strip()) for line in lines]
    communities = [[] for n in range(nsubgraphs)]
    nnodes = G.vcount()
    for n in range(nnodes):
        communities[membership[n]].append(n)
    subgraphs = [G.subgraph(communities[i]) for i in range(nsubgraphs)]

    # server id is from 0
    server_route_tables = [{} for _ in range(nsubgraphs)]
    for e in G.es:
        server_source = membership[e.source]
        server_target = membership[e.target]
        if server_source != server_target:
            if e.source in server_route_tables[server_source]:
                server_route_tables[server_source][e.source].append((e.target, server_target))
            else:
                server_route_tables[server_source][e.source] = [(e.target, server_target)]   
            if e.target in server_route_tables[server_target]:
                server_route_tables[server_target][e.target].append((e.source, server_source))
            else:
                server_route_tables[server_target][e.target] = [(e.source, server_source)]

    # write to file, server id is from 0
    for n in range(nsubgraphs):
        # generate subgraph edgelist files
        sub = nsubs + "/" + edgefile.rstrip(".txt") + ".sub" + str(n) + ".txt"
        g = subgraphs[n]

        # check graph
        if(g.is_connected()):
            print(f"Subgraph{n} is connected")
        else:
            print(f"Subgraph{n} is not connected")
            sys.exit(1)

        # node id is from 0 in subgraph
        # df = pd.DataFrame(subgraphs[n].get_edgelist())

        # node id is maintained the same as in graph
        edgelist = [(g.vs[e.source]["name"], g.vs[e.target]["name"]) for e in g.es]
        df = pd.DataFrame(edgelist)
        df.to_csv(sub, sep=" ", index=False, header=False)
        print(sub + " generated.")

        # generate subgraph edgenode route tables
        sources = list(server_route_tables[n].keys())
        targets_servers = list(server_route_tables[n].values())
        df = pd.DataFrame({'sources':sources, 'targets_servers':targets_servers})
        rt = nsubs + "/" + edgefile.rstrip(".txt") + ".rt" + str(n) + ".txt"
        df.to_csv(rt, sep=" ", index=False, header=False)
        print(rt + " generated.")

if __name__ == "__main__":
   main(sys.argv[1:])  

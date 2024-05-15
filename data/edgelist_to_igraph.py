import argparse
from ast import literal_eval
import networkx as nx
import numpy as np
import pandas as pd

##########
## assume you have airport.edges and airport.labels, output will be airport_0.edges and airport_0.labels
## e.g., python3 edgelist_to_igraph.py --data airport --weight true --labels true
## make sure you have xxx.edges when using --data, you have edge weight included in input graph edgelist, and you have yyy.labels when using --labels
##########

def parse_args():
    """
    Parse the arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Transformer")

    parser.add_argument('--data', default='blogcatalog',
                        help='Input graph edgelist, to be transformed to output graph edgelist for igraph use (contiguous node IDs starting from 0). Default is "blogcatalog". ')

    parser.add_argument('--weight', default=False,
                        help='Whether to include edge weights in output graph edgelist. Default is False.')

    parser.add_argument('--labels', default=False,
                        help='Whether to transform node labels (categories). Default is False.')

    return parser.parse_args()

def renum_nodes_contiguous(data, weight, labels):
    file = f"{data}.edges"
    file_labels = f"{data}.labels"
    file_new = f"{data}_0.edges"
    file_labels_new = f"{data}_0.labels"

    node_map = {}
    with open(file, "r") as f:
        edgelist = [line.strip().split() for line in f]
        edgelist = [list(map(int, map(float, edge))) for edge in edgelist]
        nodes = set([edge[0] for edge in edgelist] + [edge[1] for edge in edgelist])
        node_map = {node: i for i, node in enumerate(nodes)} 
        if weight:
            edgelist_new = [(node_map[edge[0]], node_map[edge[1]], edge[2]) for edge in edgelist]
        else:
            edgelist_new = [(node_map[edge[0]], node_map[edge[1]]) for edge in edgelist]
        df = pd.DataFrame(edgelist_new)
        df.to_csv(file_new, sep=" ", index=False, header=False)
    
    if labels:
        with open(file_labels, "r") as f:
            labellist = [line.strip().split() for line in f]
            labellist = [list(map(int, nodelabel)) for nodelabel in labellist]
            labellist_new = []
            for nodelabel in labellist:
                if nodelabel[0] in node_map:
                    labellist_new.append((node_map[nodelabel[0]], nodelabel[1]))
            df = pd.DataFrame(labellist_new)
            df.to_csv(file_labels_new, sep=" ", index=False, header=False)

def main():
    args = parse_args()
    renum_nodes_contiguous(args.data, args.weight, args.labels)
    print(f'Transform done.')

if __name__ == "__main__":
    main()
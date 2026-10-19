"""Read edge lists and label files into igraph / networkx graphs."""

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd


def read_edges(f):
    """Whitespace-separated edge list (source target [weight]) -> DataFrame."""
    df = pd.read_csv(f, sep=r'\s+', header=None)
    cols = ["source", "target"]
    if df.shape[1] > 2:
        cols.append("weight")
    df.columns = cols
    return df


def read_labels(f):
    """Whitespace-separated (node label) file -> DataFrame, or None if f is None."""
    if f is None:
        return None
    return pd.read_csv(f, sep=r'\s+', header=None)


def min_node_id(df):
    return np.min(np.concatenate((df["source"].values, df["target"].values)))


def build_graph(df, graph_tool):
    """Build an undirected graph. igraph requires node ids starting from 0."""
    if graph_tool == "igraph":
        if min_node_id(df) > 0:
            raise ValueError("Node ID should be from 0!")
        return ig.Graph.DataFrame(df, directed=False)
    if df.shape[1] == 2:
        return nx.from_pandas_edgelist(df, source="source", target="target")
    return nx.from_pandas_edgelist(df, source="source", target="target", edge_attr="weight")

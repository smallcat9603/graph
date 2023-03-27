import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import patches
import math
from typing import Dict, List, Tuple
import numpy as np

WIDTH = 33 / 6

LIGHT_COLORS = [
    "lightcoral",  # red
    "lightgreen",  # green
    "cyan",  # blue
    "yellow",  # yellow
    "lightgray",  # black
    "violet",  # purple
    "lime",  # green
    "lightskyblue",  # blue
    "khaki",  # yellow
    "lightsalmon",  # orange
]


def draw_graph(
        g: nx.Graph,
        filename="tmp/graph.png",
        pos=None,
        node_size=300,
        special_node: int = None,
        special_nodes: set = set(),
        special_nodes2: set = set(),
        node_color: list = None,
        node_alpha: list = None,
        edge_colors: Dict[str, List[Tuple[int]]] = None,
        edge_alpha: list = None,
        font_color: str = "k",
        bold_edge: bool = True,
        figsize=(math.sqrt(2) * WIDTH, WIDTH),
        transparent=False,
        scores: dict = None,
        font_family="DejaVu Serif"
):
    plt.figure(figsize=figsize)
    plt.axis("off")
    if pos is None:
        pos = nx.spring_layout(g, k=0.3)
    if node_color is None:
        node_color = []
        for nd in g.nodes():
            if nd == special_node:
                node_color.append("tab:red")
            elif nd in special_nodes:
                node_color.append("tab:blue")
            elif nd in special_nodes2:
                node_color.append("tab:green")
            else:
                node_color.append("tab:gray")
    if pos is None:
        pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=node_size,
        node_color=node_color,
        alpha=node_alpha
    )
    if font_color:
        nx.draw_networkx_labels(
            g,
            pos,
            font_color=font_color,
            font_family=font_family
        )
    edge_widths = []
    if bold_edge:
        for u, v, w in g.edges(data=True):
            if 'weight' in w:
                edge_widths.append(0.5 * w['weight'])
            else:
                edge_widths.append(0.5)
    else:
        edge_widths.append(0.5)

    if edge_colors:
        for edge_color, edgelist in edge_colors.items():
            nx.draw_networkx_edges(
                g,
                pos,
                edgelist=edgelist,
                width=edge_widths,
                edge_color=edge_color,
                alpha=edge_alpha
            )
    else:
        nx.draw_networkx_edges(
            g,
            pos,
            width=edge_widths,
            connectionstyle='arc3, rad = 0.05',
            alpha=edge_alpha
        )
    if scores:
        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.08)
        nx.draw_networkx_labels(g, pos_attrs, scores)
    print(filename)
    plt.savefig(filename, transparent=transparent, dpi=300)
    import subprocess
    try:
        subprocess.run(["open", filename])
    except:
        return


def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(
        G, dtype=np.bool, nodelist=node_order)
    print(adjacency_matrix)

    # Plot adjacency matrix in toned-down black and white
    fig = plt.figure(figsize=(5, 5))  # in inches
    plt.imshow(adjacency_matrix,
               cmap="Greys",
               interpolation="none")
    plt.savefig("tmp/yeah.png")

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = plt.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                           len(module),  # Width
                                           len(module),  # Height
                                           facecolor="none",
                                           edgecolor=color,
                                           linewidth="1"))
            current_idx += len(module)

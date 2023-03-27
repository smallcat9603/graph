from distutils.command.upload import upload
import community as community_louvain
from collections import defaultdict
import networkx as nx
import subprocess
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from image import concatanate_images, upload_to_imgbb
from read_graph import read_graph
import networkx.algorithms.community as nx_comm
from louvain import louvain_communities as louvain
from charts import *


def get_gname(file_path):
    if file_path[:5] == "graph":
        return file_path[6:file_path.rfind(".")]
    elif file_path[:15] == "processed_graph":
        return file_path[16:file_path.rfind(".")]
    else:
        print("file_path error")
        exit(0)


def show_graph(g, partition=None):
    if partition:
        # draw the graph
        pos = nx.spring_layout(g)
        # color the nodes according to their partition
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        nx.draw_networkx_nodes(g, pos, partition.keys(), node_size=40,
                               cmap=cmap, node_color=list(partition.values()))
        nx.draw_networkx_edges(g, pos, alpha=0.5)
        nx.draw_networkx_labels(g, pos)
    else:
        nx.draw(g, with_labels=True)
    filepath = "img/tmp.pdf"
    plt.savefig(filepath)
    subprocess.call(('open', filepath))


def output_info_table(graph_names):
    # print("|graph|n|m|average degree|edge density|# communities by Louvain|modularity of best partition|")
    print("|graph|n|m|average degree|edge density|")
    # print("|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    print("|:--:|:--:|:--:|:--:|:--:|")
    for graph_name in graph_names:
        print("|", end='')
        print(get_gname(graph_name), end='|')
        g = read_graph(graph_name)
        print(g.number_of_nodes(), end='|')
        print(g.number_of_edges(), end='|')
        print(g.number_of_edges()/g.number_of_nodes(), end='|')
        print(nx.classes.function.density(g), end='|')
        # communities = louvain(g)
        # print(len(communities), end='|')
        # c = list(set(c) for c in communities.values())
        # print(nx_comm.modularity(g, c), end='|')
        print()
    print()


def output_degree_distribution(graph_names):
    filenames = []
    for graph_name in graph_names:
        gname = get_gname(graph_name)
        g = read_graph(graph_name)
        filename = f"tmp/degree-distribution-{gname}.png"
        degree_distribution(g, gname, filename=filename)
        filenames.append(filename)
    filename = concatanate_images(filenames, "tmp/degree-distribution", 3, 2)
    upload_to_imgbb(filename)


def print_info(g):
    print("n " + str(g.number_of_nodes()), end=', ')
    print("m: " + str(g.number_of_edges()), end=', ')
    print("average_degree: " + str(g.number_of_edges()/g.number_of_nodes()), end=', ')
    print("density: " + str(nx.classes.function.density(g)), end=', ')
    partition, communities = louvain(g)
    print("num_communities: " + str(len(communities)))


def degree_distribution(g: nx.Graph, memo: str = None, filename: str = "tmp/degree_distribution.png"):
    degrees = [g.degree(nd) for nd in g]
    from collections import Counter
    c = Counter(degrees)
    x, y = [], []
    for deg, cnt in c.items():
        x.append(deg)
        y.append(cnt/g.number_of_nodes())
    draw_scatter(
        [x],
        [y],
        x_axis_title="degree",
        y_axis_title="frequency",
        title=f"degree distribution ({memo})",
        xscale="log",
        yscale="log",
        left=0.99,
        bottom=min(y) * 0.99,
        filename=filename,
    )

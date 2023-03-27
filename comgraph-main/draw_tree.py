from graphviz import Graph
import subprocess


def draw_tree(edges, graph_name, option=None):
    G = Graph(format="pdf")
    G.attr("node", shape="box")
    filepath = "img/" + graph_name
    if option:
        filepath += "-" + option
    print(filepath + ".pdf")
    for i, j in edges:
        G.edge(i, j)
    G.render(filepath)
    subprocess.run(["open", filepath + ".pdf"])

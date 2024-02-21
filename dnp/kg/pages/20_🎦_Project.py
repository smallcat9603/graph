import streamlit as st
from pages.lib import cypher, flow

if 'data' not in st.session_state:
   st.title("No Graph Data")
   st.warning("You should load graph data first!", icon='⚠')
   st.stop()
else:
   st.title(f"Dataset {st.session_state['data']} Projection")

st.divider()

##### List in-memory graph

st.header("In-memory graph list")

graph_ls = st.session_state["gds"].graph.list()["graphName"]
if len(graph_ls) > 0:
    for g in graph_ls:
        G = st.session_state["gds"].graph.get(g)
        st.success(f"Graph {g}: {G.node_count()} nodes and {G.relationship_count()} relationships")  
        st.subheader("Graph Statistics (project)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("# Nodes", str(G.node_count()))
        col2.metric("# Edges", str(G.relationship_count()))
        col3.metric("Density", str(G.density()))
        col4.metric("Memory", str(G.memory_usage())) 
else:
    st.warning('There are currently no graphs in memory.')

##### Create in-memory graphs & Drop in-memory graph

def prj_graph(node_properties, relationship_properties):
    exists_result = st.session_state["gds"].graph.exists(st.session_state["graph_name"])
    if exists_result["exists"]:
        G = st.session_state["gds"].graph.get(st.session_state["graph_name"])
        G.drop()
    G, result = st.session_state["gds"].graph.project(st.session_state["graph_name"], node_properties, relationship_properties)

def drop_graph(drop_g):
    if drop_g is not None:
        flow.drop_memory_graph(drop_g)

st.header("Create in-memory graph")

nodes = cypher.get_node_labels()
relationships = cypher.get_relationship_types()

node_labels = st.multiselect("Node labels", nodes, nodes)
node_properties = {}
for node_label in node_labels:
    properties = cypher.get_node_properties(node_label)
    node_properties[node_label] = {"properties": properties}
    st.caption(f"{node_label}: {', '.join(properties)}")

relationship_types = st.multiselect("Relationship types", relationships, relationships)
relationship_properties = {}
for relationship_type in relationship_types:
    properties = cypher.get_relationship_properties(relationship_type)
    relationship_properties[relationship_type] = {"orientation": "UNDIRECTED", "properties": properties}
    st.caption(f"{relationship_type}: {', '.join(properties)}")

st.button("Create in-memory graph", type="secondary", on_click=prj_graph, args=(node_properties, relationship_properties))
  
st.header("Drop in-memory graph")

drop_g = st.selectbox('Choose an graph to drop: ', st.session_state["gds"].graph.list()["graphName"])

st.button("Drop in-memory graph", type="secondary", on_click=drop_graph, args=(drop_g,))

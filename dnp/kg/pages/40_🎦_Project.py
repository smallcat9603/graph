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
else:
    st.warning('There are currently no graphs in memory.')

##### Create in-memory graphs & Drop in-memory graph

def prj_graph(node_label_list, relationship_type, relationship_property_list):
    G, result = flow.project_graph(node_label_list, relationship_type, relationship_property_list)

def drop_graph(drop_g):
    if drop_g is not None:
        flow.drop_memory_graph(drop_g)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Create in-memory graph")

    nodes = cypher.get_node_labels()
    relationships = cypher.get_relationship_types()

    node_label_list = st.multiselect("Node labels", nodes, nodes)
    relationship_type = st.selectbox("Relationship type", relationships)
    relationship_properties = cypher.get_relationship_properties(relationship_type)
    relationship_property_list = st.multiselect("Relationship properties", relationship_properties, relationship_properties)

    st.button("Create in-memory graph", type="secondary", on_click=prj_graph, args=(node_label_list, relationship_type, relationship_property_list))
  
with col2:
    st.subheader("Drop in-memory graph")

    drop_g = st.selectbox('Choose an graph to drop: ', st.session_state["gds"].graph.list()["graphName"])

    st.button("Drop in-memory graph", type="secondary", on_click=drop_graph, args=(drop_g,))

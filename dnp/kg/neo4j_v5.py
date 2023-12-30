from graphdatascience import GraphDataScience
import sys
import streamlit as st

##############################
### neo4j desktop v5.11.0 ###
##############################

if "reboot" not in st.session_state:
   st.session_state["reboot"] = False
if st.session_state["reboot"] == True:
   st.stop()

st.title("Graph Data App")

filename = __file__.split("/")[-1]
if filename.startswith("neo4j"):
    # desktop
    host = "bolt://localhost:7687"
    user = "neo4j"
    password= "j4oenj4oen"
elif filename.startswith("app"):
    # sandbox
    host = "bolt://3.228.13.111:7687"
    user = "neo4j"
    password= "centers-operators-tips"
st.session_state["gds"] = GraphDataScience(host, auth=(user, password))

st.header("gds version")
st.write(st.session_state["gds"].version())

st.session_state["graph_name"] = "testgraph" # project graph name

@st.cache_data
def cypher(query):
   return st.session_state["gds"].run_cypher(query)

##############################
### free up memory ###
##############################

def free_up_memory():
    exists_result = st.session_state["gds"].graph.exists(st.session_state["graph_name"])
    if exists_result["exists"]:
        G = st.session_state["gds"].graph.get(st.session_state["graph_name"])
        G.drop()    
    query = """
    MATCH (n) DETACH DELETE n
    """
    cypher(query)
    st.session_state["gds"].close()
    st.write("gds closed")
    st.session_state["reboot"] = True

st.button("Free up memory", type="primary", on_click=free_up_memory) 

##############################
### parameters ###
##############################

st.session_state["KEY"] = "AIzaSyAPQNUpCCFrsJhX2A-CgvOG4fDWlxuA8ec" # api key

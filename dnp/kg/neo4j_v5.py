from graphdatascience import GraphDataScience
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
    host = st.secrets["NEO4J_LOCAL"]
    user = st.secrets["NEO4J_LOCAL_USER"]
    password = st.secrets["NEO4J_LOCAL_PASSWORD"]
elif filename.startswith("app"):
    # sandbox
    host = st.secrets["NEO4J_SANDBOX"]
    user = st.secrets["NEO4J_SANDBOX_USER"]
    password = st.secrets["NEO4J_SANDBOX_PASSWORD"]
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
    st.header("gds closed")
    st.info("Click 'Clear cache (C)' and 'Rerun (R)'")
    st.session_state["reboot"] = True

st.button("Free up memory", type="primary", on_click=free_up_memory) 

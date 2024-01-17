from graphdatascience import GraphDataScience
import streamlit as st

##############################
### neo4j desktop v5.11.0 ###
##############################

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

st.success(f"Connection successful to GDBS server: {host}") 
st.info(f"GDS version: {st.session_state['gds'].version()}")
st.session_state["graph_name"] = "testgraph" # project graph name
st.divider()
st.title("GDBS Status")
reset_disabled = False
container_data = st.container(border=False)
if "data" not in st.session_state:
    container_data.success("Database is empty. Now you can load graph data!")
    reset_disabled = True
else:
    container_data.warning(f"Data {st.session_state['data']} is loaded. When switching between graph databases, 'Reset' the GDBS server status first!")

@st.cache_data
def cypher(query):
   return st.session_state["gds"].run_cypher(query)

if st.button("Reset", type="primary", disabled=reset_disabled):
    exists_result = st.session_state["gds"].graph.exists(st.session_state["graph_name"])
    if exists_result["exists"]:
        G = st.session_state["gds"].graph.get(st.session_state["graph_name"])
        G.drop()    
    query = """
    MATCH (n) DETACH DELETE n
    """
    cypher(query)
    st.cache_data.clear() # clear cache data via @st.cache_data, not including st.session_state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()
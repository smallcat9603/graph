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
st.divider()
st.caption("When switching between graph databases, 'Reset' the GDBS server status first!")

st.session_state["graph_name"] = "testgraph" # project graph name

@st.cache_data
def cypher(query):
   return st.session_state["gds"].run_cypher(query)

if st.button("Reset", type="primary"):
    exists_result = st.session_state["gds"].graph.exists(st.session_state["graph_name"])
    if exists_result["exists"]:
        G = st.session_state["gds"].graph.get(st.session_state["graph_name"])
        G.drop()    
    query = """
    MATCH (n) DETACH DELETE n
    """
    cypher(query)
    st.cache_data.clear() # clear cache data via @st.cache_data, not including st.session_state
    st.success("Cache cleared! Now you can load graph data!")
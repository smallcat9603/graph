from graphdatascience import GraphDataScience
import streamlit as st
from pages.lib import cypher, flow

##############################
### neo4j desktop v5.11.0 ###
##############################

st.markdown("""
            **Author:** smallcat (huyao0107@gmail.com)

            © 2024 Keio Univ. All rights reserved.
""")

st.title(":cat2: Welcome to Graph Data App!")

filename = __file__.split("/")[-1]
if filename.startswith("neo4j"):
    # desktop
    neo4j_server = "NEO4J_LOCAL"
elif filename.startswith("App"):
    # sandbox
    neo4j_server = "NEO4J_SANDBOX"

st.session_state["hp"] = filename
st.session_state["host"] = st.secrets[neo4j_server]
st.session_state["user"] = st.secrets[neo4j_server+"_USER"]
st.session_state["password"] = st.secrets[neo4j_server+"_PASSWORD"]
st.session_state["gds"] = GraphDataScience(st.session_state["host"], auth=(st.session_state["user"], st.session_state["password"]))

st.session_state["dir"] = "https://raw.githubusercontent.com/smallcat9603/graph/main/dnp/kg/data/"

st.success(f"Connection successful to GDBS server: {st.session_state['host']}") 
st.info(f"GDS version: {st.session_state['gds'].version()}")
st.session_state["graph_name"] = "testgraph" # project graph name
st.divider()
st.title("GDBS Status")
container_data = st.container(border=False)
if "data" not in st.session_state:
    container_data.success("Database is empty. Now you can load graph data!")
else:
    container_data.warning(f"Data {st.session_state['data']} is loaded. When switching between graph databases, 'Reset' the GDBS server status first!")

if st.button("Reset", type="primary"):
    flow.drop_memory_graph(st.session_state["graph_name"])
    cypher.free_up_db()

    st.cache_data.clear() # clear cache data via @st.cache_data, not including st.session_state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()
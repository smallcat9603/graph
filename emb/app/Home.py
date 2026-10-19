import streamlit as st

from ui import core

##############################
### neo4j desktop 2.1.3 (2026.02.2) ###
##############################

st.markdown("""
            **Author:** smallcat (huyao0107@gmail.com)

            © 2024 Keio Univ. All rights reserved.
""")

st.title(":cat2: Welcome to Graph Data App!")

if "gds" not in st.session_state:
    core.connect()

st.success(f"Connection successful to GDBS server: {st.session_state['host']}")
st.info(f"GDS version: {core.gds().version()}")
st.divider()
st.title("GDBS Status")
container_data = st.container(border=False)
if "data" not in st.session_state:
    container_data.success("Database is empty. Now you can load graph data!")
else:
    container_data.warning(f"Data {st.session_state['data']} is loaded. When switching between graph databases, 'Reset' the GDBS server status first!")

if st.button("Reset", type="primary"):
    core.reset()
    st.rerun()

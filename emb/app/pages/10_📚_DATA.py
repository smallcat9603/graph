import streamlit as st

from embkg import datasets
from ui import views

st.title("Dataset Graph Construction")

TYPE, DATA, LANGUAGE = views.select_data()

if TYPE == "CYPHER":
    views.construct_graph_cypherfile(DATA)
else:
    params = views.set_param(DATA)
    DATA_TYPE = params[1]
    if TYPE == "DNP":
        DATA_URL, QUERY_DICT = datasets.dnp_source(DATA_TYPE)
    else:
        DATA_URL, QUERY_DICT = datasets.wiki_source(DATA)
    views.construct_graph_cypher(DATA, LANGUAGE, DATA_URL, QUERY_DICT, *params)

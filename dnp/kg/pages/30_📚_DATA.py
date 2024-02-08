import streamlit as st
import os
from pages.lib import flow

st.title(f"Dataset Graph Construction")

TYPE, DATA, LANGUAGE = flow.select_data()

if TYPE == "DNP":

    nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY, WORD_CLASS, PIPELINE_SIZE = flow.set_param(DATA)

    QUERY_DICT = {} # query dict {QUERY_NAME: QUERY_URL}
    if DATA_TYPE == "TXT":
        DATA_URL = os.path.dirname(os.path.dirname(__file__)) + "/data/newsrelease_B-1-100_C-1-4/"
        QUERY_DICT["C-1"] = DATA_URL + "C-1.txt"
        QUERY_DICT["C-2"] = DATA_URL + "C-2.txt"
        QUERY_DICT["C-3"] = DATA_URL + "C-3.txt"
        QUERY_DICT["C-4"] = DATA_URL + "C-4.txt"
    elif DATA_TYPE == "URL":
        DATA_URL = f"{st.session_state['dir']}articles.csv"
        QUERY_DICT["C-1"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_1.html"
        QUERY_DICT["C-2"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_2.html"
        QUERY_DICT["C-3"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_3.html"
        QUERY_DICT["C-4"] = "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231003_1.html" 

    flow.construct_graph_cypher(DATA, LANGUAGE, DATA_URL, QUERY_DICT, nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY, WORD_CLASS, PIPELINE_SIZE)

elif TYPE == "WIKI":

    nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY, WORD_CLASS, PIPELINE_SIZE = flow.set_param(DATA)

    QUERY_DICT = {} # query dict {QUERY_NAME: QUERY_URL}
    if DATA == "FP100":
        DATA_URL = f"{st.session_state['dir']}wikidata_footballplayer_100.csv"
        QUERY_DICT["Thierry Henry"] = "https://en.wikipedia.org/wiki/Thierry_Henry"
    elif DATA == "P100":
        DATA_URL = f"{st.session_state['dir']}wikidata_persons_100.csv"  
        QUERY_DICT["Joe Biden"] = "https://en.wikipedia.org/wiki/Joe_Biden"
    elif DATA == "P1000":
        DATA_URL = f"{st.session_state['dir']}wikidata_persons_1000.csv"  
        QUERY_DICT["Joe Biden"] = "https://en.wikipedia.org/wiki/Joe_Biden"
    elif DATA == "P10000":
        DATA_URL = f"{st.session_state['dir']}wikidata_persons_10000.csv"  
        QUERY_DICT["Joe Biden"] = "https://en.wikipedia.org/wiki/Joe_Biden"

    flow.construct_graph_cypher(DATA, LANGUAGE, DATA_URL, QUERY_DICT, nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY, WORD_CLASS, PIPELINE_SIZE)

elif TYPE == "CYPHER":

    flow.construct_graph_cypherfile(DATA, LANGUAGE)

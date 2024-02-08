import streamlit as st
import os
import re
import time
from pages.lib import cypher, flow

##############################
### Setting param ###
##############################

DATA = __file__.split("/")[-1].split(".")[0].split("_")[-1]
LANGUAGE = "ja"

st.title(f"{DATA} Dataset")
st.info("This database includes 100 DNP newsreleases, and 4 Toppan newsreleases.")
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

##############################
### Import CSV ###
##############################

cypher.create_constraint(st.session_state["constraint"])
if DATA_LOAD == "Offline":
    result_import_graph_data = cypher.import_graph_data(DATA)

##############################
### Create Article-[Noun]-Article Graph ###
##############################

st.divider()
st.title("Progress")
progress_bar = st.progress(0, text="Initialize...")
start_time = time.perf_counter()
container_status = st.container(border=False)

##############################
### create url nodes (article, person, ...) ###
##############################

progress_bar.progress(20, text="Create url nodes...")

if DATA_LOAD != "Offline":
    if DATA_TYPE == "TXT":
        for idx in range(1, 101):
            node = "B-" + str(idx)
            file = DATA_URL + node + ".txt"
            content = ""
            with open(file, 'r') as f:
                content = f.read()
                content = re.sub('\n+', ' ', content)
            query = f"""
            MERGE (a:Article {{ name: "{node}", url: "{file}", body: "{content}" }})
            """
            cypher.run(query)
        # query
        for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
            content = ""
            with open(QUERY_URL, 'r') as f:
                content = f.read()
                content = re.sub('\n+', ' ', content)
            query = f"""
            MERGE (q:Query {{ name: "{QUERY_NAME}", url: "{QUERY_URL}", body: "{content}" }})
            """
            cypher.run(query)
    else:
        cypher.load_data_url(DATA_URL)
        # query
        for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
            cypher.create_query_node(QUERY_NAME, QUERY_URL)

##############################
### set phrase and salience properties ###
##############################
  
progress_bar.progress(40, text="Set phrase and salience properties...")

if DATA_LOAD == "Semi-Online":
    result_set_phrase_salience_properties_csv = cypher.set_phrase_salience_properties_csv(f"{st.session_state['dir']}{DATA}.csv")
    cypher.set_phrase_salience_properties_csv(f"{st.session_state['dir']}{DATA}.csv", query_node=True)
elif DATA_LOAD == "Online":
    result_set_phrase_salience_properties_gcp = cypher.set_phrase_salience_properties_gcp(GCP_API_KEY)
    cypher.set_phrase_salience_properties_gcp(GCP_API_KEY, query_node=True)
elif DATA_LOAD == "On-the-fly":
    result_set_phrase_salience_properties_spacy = cypher.set_phrase_salience_properties_spacy(LANGUAGE, WORD_CLASS, PIPELINE_SIZE, nphrase, query_node=False)
    cypher.set_phrase_salience_properties_spacy(LANGUAGE, WORD_CLASS, PIPELINE_SIZE, nphrase, query_node=True)

##############################
### create noun-url relationships ###
##############################

progress_bar.progress(60, text="Create noun-url relationships...")

if DATA_LOAD != "Offline":
    cypher.create_noun_article_relationships(nphrase)
    cypher.create_noun_article_relationships(nphrase, query_node=True)
    
##############################
### create article-article relationships ###
##############################

progress_bar.progress(80, text="Create article-article relationships...")

if DATA_LOAD != "Offline":
    cypher.create_article_article_relationships(nphrase)
    cypher.create_article_article_relationships(nphrase, query_node=True)

##############################
### state update ###
##############################
    
flow.update_state(DATA, DATA_LOAD, QUERY_DICT)

##############################
### export to csv in import/ ###
##############################

progress_bar.progress(100, text="Finished. Show graph statistics...")

end_time = time.perf_counter()
execution_time_ms = (end_time - start_time) * 1000
container_status.success(f"Loading finished: {execution_time_ms:.1f} ms. Graph data can be queried.")

st.divider()
flow.show_graph_statistics()

st.caption("Save graph data including nodes and edges into csv files")
if st.button("Save graph data (.csv)"):
    cypher.save_graph_data(DATA)

##############################
### Verbose ###
##############################

with st.expander("Debug Info"):
    st.header("Data Source")
    st.write(DATA_URL)
    st.header("Query Dict")
    st.table(QUERY_DICT)

    if DATA_LOAD == "Offline":
        st.info("Importing nodes and relationships from csv files finished")
        st.write(result_import_graph_data)

    if DATA_LOAD == "Semi-Online":
        st.header("set phrase salience properties (csv)")
        st.write(result_set_phrase_salience_properties_csv)
    elif DATA_LOAD == "Online":
        st.header("set phrase salience properties (gcp)")
        st.write(result_set_phrase_salience_properties_gcp)
    elif DATA_LOAD == "On-the-fly":
        st.header("set phrase salience properties (spacy)")
        st.write(result_set_phrase_salience_properties_spacy)

"""Reusable Streamlit widgets: dataset forms, graph construction flow, plots."""

import time

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import optuna
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from embkg import cypher_ops, datasets, evaluate
from ui.core import run


def select_data():
    TYPE = st.radio("Select one data type:", list(datasets.DATASETS), horizontal=True)
    options = datasets.DATASETS[TYPE]
    DATA = st.radio("Select one dataset", list(options), captions=list(options.values()))
    return TYPE, DATA, datasets.LANGUAGE[TYPE]


def set_param(DATA):
    st.title("Parameters")
    form = st.form("parameters")
    nphrase = form.slider("Number of nouns extracted from each article (50 if Offline is selected)",
                          1,
                          100,
                          50)
    if DATA == "DNP":
        DATA_TYPE = form.radio("Data type",
                               ["TXT", "URL"],
                               horizontal=True,
                               captions=["currently used only for dnp data", "parse html to retrive content"])
    else:
        DATA_TYPE = form.radio("Data type",
                               ["URL"],
                               horizontal=True,
                               captions=["parse html to retrive content"])
    # offline opt: neo4j-admin database dump/load, require to stop neo4j server
    DATA_LOAD = form.radio("Data load",
                           ["Offline", "Semi-Online", "Online", "On-the-fly"],
                           horizontal=True,
                           captions=["load nodes and relationships from local (avoid to use gcp api, very fast)", "load nodes from local and create relationships during runtime (avoid to use gcp api, fast)", "create nodes and relationships during runtime (use gcp api, slow)", "use spaCy to extract keywords from each article (free)"],
                           index=0)
    col1, col2 = form.columns(2)
    expander_gcp_api_key = col1.expander("GCP API Key (Mandatory for Online)")
    GCP_API_KEY = expander_gcp_api_key.text_input("GCP API Key",
                                type="password",
                                placeholder="should not be empty for Online",
                                label_visibility="collapsed")
    expander_keywords = col2.expander("Keywords (Optional for On-the-fly)")
    WORD_CLASS = expander_keywords.multiselect("Keywords",
                                             ["NOUN", "ADJ", "VERB"],
                                             ["NOUN"])
    PIPELINE_SIZE = expander_keywords.radio("Pipeline size",
                           ["Small", "Medium", "Large"],
                           horizontal=True,
                           captions=["10MB+", "40MB+", "500MB+"])

    run_disabled = False
    if "data" in st.session_state and st.session_state["data"] != DATA:
        run_disabled = True
        form.warning("Please 'Reset' the database status first before you 'Run'!", icon='⚠')

    if form.form_submit_button("Run", type="primary", disabled=run_disabled):
        if DATA_LOAD == "Online" and GCP_API_KEY == "":
            form.warning("Please input GCP API Key (Mandatory for Online) before you 'Run'!", icon='⚠')
            st.stop()
        elif DATA_LOAD == "On-the-fly" and len(WORD_CLASS) == 0:
            form.warning("Please select at least one keyword type before you 'Run'!", icon='⚠')
            st.stop()
    else:
        if "data" not in st.session_state or st.session_state["data"] != DATA:
            st.stop()

    return nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY, WORD_CLASS, PIPELINE_SIZE


def show_graph_statistics():
    st.title("Graph Statistics")
    result = cypher_ops.get_graph_statistics(run)

    col1, col2 = st.columns(2)
    col1.metric("# Nodes", result["nodeCount"][0])
    col2.metric("# Edges", result["relCount"][0])

    col1, col2 = st.columns(2)
    with col1.expander("Node Labels"):
        st.table(result["labels"][0])
    with col2.expander("Relationship Types"):
        st.table(result["relTypesCount"][0])


def save_graph_button(DATA):
    st.caption("Save graph data including nodes and edges into csv files")
    if st.button("Save graph data (.csv)"):
        st.write(cypher_ops.save_graph_data(run, DATA))


def construct_graph_cypher(DATA, LANGUAGE, DATA_URL, QUERY_DICT, nphrase, DATA_TYPE, DATA_LOAD, GCP_API_KEY, WORD_CLASS, PIPELINE_SIZE):
    """Build the Article-[Noun]-Article graph (DNP and WIKI datasets)."""
    cypher_ops.create_constraint(run, "ID_UNIQUE")
    if DATA_LOAD == "Offline":
        result_import_graph_data = cypher_ops.import_graph_data(run, st.session_state["dir"], DATA)

    st.divider()
    st.title(f"Progress ({DATA})")
    progress_bar = st.progress(0, text="Initialize...")
    start_time = time.perf_counter()
    container_status = st.container(border=False)

    progress_bar.progress(20, text="Create url nodes...")
    if DATA_LOAD != "Offline":
        if DATA_TYPE == "TXT":
            cypher_ops.load_articles_txt(run, DATA_URL, QUERY_DICT)
        else:
            cypher_ops.load_data_url(run, DATA_URL)
            for QUERY_NAME, QUERY_URL in QUERY_DICT.items():
                cypher_ops.create_query_node(run, QUERY_NAME, QUERY_URL)

    progress_bar.progress(40, text="Set phrase and salience properties...")
    if DATA_LOAD == "Semi-Online":
        file = f"{st.session_state['dir']}{DATA}.csv"
        result_phrase = cypher_ops.set_phrase_salience_properties_csv(run, file)
        cypher_ops.set_phrase_salience_properties_csv(run, file, query_node=True)
    elif DATA_LOAD == "Online":
        result_phrase = cypher_ops.set_phrase_salience_properties_gcp(run, GCP_API_KEY)
        cypher_ops.set_phrase_salience_properties_gcp(run, GCP_API_KEY, query_node=True)
    elif DATA_LOAD == "On-the-fly":
        result_phrase = cypher_ops.set_phrase_salience_properties_spacy(run, LANGUAGE, WORD_CLASS, PIPELINE_SIZE, nphrase)
        cypher_ops.set_phrase_salience_properties_spacy(run, LANGUAGE, WORD_CLASS, PIPELINE_SIZE, nphrase, query_node=True)

    progress_bar.progress(60, text="Create noun-url relationships...")
    if DATA_LOAD != "Offline":
        cypher_ops.create_noun_article_relationships(run, nphrase)
        cypher_ops.create_noun_article_relationships(run, nphrase, query_node=True)

    progress_bar.progress(80, text="Create article-article relationships...")
    if DATA_LOAD != "Offline":
        cypher_ops.create_article_article_relationships(run, nphrase)
        cypher_ops.create_article_article_relationships(run, nphrase, query_node=True)

    st.session_state["data"] = DATA
    st.session_state["load"] = DATA_LOAD
    st.session_state["query"] = QUERY_DICT

    progress_bar.progress(100, text="Finished. Show graph statistics...")
    execution_time_ms = (time.perf_counter() - start_time) * 1000
    container_status.success(f"Loading finished: {execution_time_ms:.1f} ms. Graph data can be queried.")

    st.divider()
    show_graph_statistics()
    save_graph_button(DATA)

    with st.expander("Debug Info"):
        st.header("Data Source")
        st.write(DATA_URL)
        st.header("Query Dict")
        st.table(QUERY_DICT)

        if DATA_LOAD == "Offline":
            st.info("Importing nodes and relationships from csv files finished")
            st.write(result_import_graph_data)
        else:
            st.header(f"set phrase salience properties ({DATA_LOAD})")
            st.write(result_phrase)


def construct_graph_cypherfile(DATA):
    """Load a CYPHER dataset by running its .cypher file."""
    run_disabled = False
    if "data" in st.session_state and st.session_state["data"] != DATA:
        run_disabled = True
        st.warning("Please 'Reset' the database status first before you 'Run'!", icon='⚠')

    if st.button("Run", type="primary", disabled=run_disabled):
        cypher_ops.run_file(run, datasets.CYPHER_FILES[DATA])
        st.session_state["data"] = DATA
    else:
        if "data" not in st.session_state or st.session_state["data"] != DATA:
            st.stop()

    st.success(f"Dataset {DATA} is loaded.")
    show_graph_statistics()
    save_graph_button(DATA)


def plot_similarity(result, query_node, similarity_method, limit):
    fig, ax = plt.subplots()
    articles = result["Article"]
    y_pos = np.arange(len(articles))

    ax.barh(y_pos, result["Similarity"])
    ax.set_yticks(y_pos, labels=articles)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Similarity Score')
    ax.set_title(f'Similarity to {query_node} by {similarity_method} (Top-{limit})')

    st.pyplot(fig)


def chart_and_table(result, query_node, method, limit):
    tab01, tab02 = st.tabs(["Chart", "Table"])
    with tab01:
        plot_similarity(result, query_node, method, limit)
    with tab02:
        st.write(result)


def plot_tsne_alt(result):
    tsne_df = evaluate.tsne_frame(result)
    chart = alt.Chart(tsne_df).mark_circle(size=60).encode(
        x="x",
        y="y",
        color="category",
        tooltip=["name", "category"],
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)


def modeler(result):
    """Show SVM accuracy over 5 random splits and the last split's confusion matrix."""
    acc_scores, y_test, pred, classes = evaluate.svm_accuracy(result)
    st.write('Accuracy scores: ', acc_scores)
    st.write('Mean accuracy: ', np.mean(acc_scores))

    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, pred, labels=classes, normalize="true")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(ax=ax)
    st.pyplot(fig)


def show_tuning_result(study):
    st.write(f"Accuracy: {study.best_trial.value}")
    st.write(f"Best hyperparameters: {study.best_trial.params}")
    st.plotly_chart(optuna.visualization.plot_optimization_history(study))
    st.plotly_chart(optuna.visualization.plot_param_importances(study))


def tune(objective, initial_params, n_trials=100):
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(initial_params)
    study.optimize(objective, n_trials=n_trials)
    show_tuning_result(study)


# "UI Interaction" tabs

def interact_knn(emb, QUERY_DICT):
    col1, col2 = st.columns(2)
    with col1:
        query_node = st.selectbox("Query node", tuple(QUERY_DICT.keys()), key=f"{emb}_col1")
    with col2:
        limit = st.selectbox("Limit", ("5", "10", "15", "20"), key=f"{emb}_col2")
    st.write("The top-" + limit + " similar nodes for query " + query_node + " are ranked as follows (" + emb + ")")
    query = cypher_ops.knn_query(emb, query_node, limit)
    chart_and_table(run(query), query_node, emb, limit)
    return query


def interact_node_similarity(QUERY_DICT):
    col1, col2, col3 = st.columns(3)
    with col1:
        query_node = st.selectbox("Query node", tuple(QUERY_DICT.keys()))
    with col2:
        similarity_method = st.selectbox("Similarity method", ("JACCARD", "OVERLAP", "COSINE", "PPR"))
    with col3:
        limit = st.selectbox("Limit", ("5", "10", "15", "20"))
    st.write("The top-" + limit + " similar nodes for query " + query_node + " are ranked as follows (" + similarity_method + ")")
    query = cypher_ops.node_similarity_query(query_node, similarity_method, limit)
    chart_and_table(run(query), query_node, similarity_method, limit)
    return query


def interact_multiple_queries(QUERY_DICT):
    col1, col2, col3 = st.columns(3)
    with col1:
        queries = list(QUERY_DICT.keys())
        query_nodes = st.multiselect("Query node", queries, queries[:2])
    with col2:
        similarity_method = st.selectbox("Similarity method", ("JACCARD", "OVERLAP", "COSINE", "PPR"), key="sm")
    with col3:
        limit = st.selectbox("Limit", ("5", "10", "15", "20"), key="lim")
    st.write("The top-" + limit + " similar nodes for queries " + ', '.join(query_nodes) + " are ranked as follows (" + similarity_method + ")")
    query = cypher_ops.multiple_queries_query(query_nodes, similarity_method, limit)
    chart_and_table(run(query), ', '.join(query_nodes), similarity_method, limit)
    return query


def interact_related_articles():
    noun = st.text_input("Keyword", "環境")
    query = cypher_ops.related_articles_query(noun)
    st.write(run(query))
    return query


def interact_fixed(query):
    st.write(run(query))
    return query

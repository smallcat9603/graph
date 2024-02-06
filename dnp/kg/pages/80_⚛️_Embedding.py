import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from pages.lib import cypher, flow


if 'data' not in st.session_state:
   st.title("No Graph Data")
   st.warning("You should load graph data first!", icon='⚠')
   st.stop()
else:
   st.title(f"Dataset {st.session_state['data']} Embedding Visualizer")

st.divider()

##### Create in-memory graphs & Drop in-memory graph

def prj_graph(node_label_list, relationship_type, relationship_property_list):
    G, result = flow.project_graph(node_label_list, relationship_type, relationship_property_list)

def drop_graph(drop_g):
    if drop_g is not None:
        flow.drop_memory_graph(drop_g)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Create in-memory graph")

    nodes = cypher.get_node_labels()
    relationships = cypher.get_relationship_types()

    node_label_list = st.multiselect("Node labels", nodes, nodes)
    relationship_type = st.selectbox("Relationship type", relationships)
    relationship_properties = cypher.get_relationship_properties(relationship_type)
    relationship_property_list = st.multiselect("Relationship properties", relationship_properties, relationship_properties)

    st.button("Create in-memory graph", type="secondary", on_click=prj_graph, args=(node_label_list, relationship_type, relationship_property_list))
  
with col2:
    st.subheader("Drop in-memory graph")

    drop_g = st.selectbox('Choose an graph to drop: ', st.session_state["gds"].graph.list()["graphName"])

    st.button("Drop in-memory graph", type="secondary", on_click=drop_graph, args=(drop_g,))


##### List in-memory graph

st.divider()
st.header("In-memory graph list")

graph_ls = st.session_state["gds"].graph.list()["graphName"]
if len(graph_ls) > 0:
    for g in graph_ls:
        G = st.session_state["gds"].graph.get(g)
        st.info(f"Graph {g}: {G.node_count()} nodes and {G.relationship_count()} relationships")  
else:
    st.info('There are currently no graphs in memory.')
    st.stop()

##############################
#
#   Embedding
#
##############################

st.divider()
st.header("Embedding")

emb_graph = st.selectbox('Enter graph name for embedding creation: ', graph_ls)
G = st.session_state["gds"].graph.get(emb_graph)

rprop = [None]
for rt in G.relationship_types():
    rprop += G.relationship_properties(rt)
nprop = []
for nl in G.node_labels():
    nprop += G.node_properties(nl)

##### FastRP embedding creation

# Description of hyperparameters can be found: https://neo4j.com/docs/graph-data-science/current/algorithms/fastrp/#algorithms-embeddings-fastrp
with st.expander('FastRP embedding creation'):
    frp_seed = st.slider('Random seed', value=42, min_value=1, max_value=99)
    frp_dim = st.select_slider("Embedding dimenson", value=256, options=[128, 256, 512, 1024], key="frp_dim")
    frp_norm = st.slider("Normalization strength", value=0., min_value=-1., max_value=1.)
    frp_it_weight1 = st.slider("Iteration weight 1", value=0., min_value=0., max_value=1.)
    frp_it_weight2 = st.slider("Iteration weight 2", value=1., min_value=0., max_value=1.)
    frp_it_weight3 = st.slider("Iteration weight 3", value=1., min_value=0., max_value=1.)
    node_self_infl = st.slider("Node self influence", value=0., min_value=0., max_value=1.)
    rel_weight_prop = st.selectbox("Relationship weight property", rprop, key="frp_rwp")
    prop_rat = st.slider("Property ratio", value=0., min_value=0., max_value=1.)
    feat_prop = st.selectbox("Feature properties", nprop)

    if st.button("Create FastRP embedding"):
        st.session_state["gds"].fastRP.write(
            G,
            embeddingDimension=frp_dim,
            normalizationStrength=frp_norm,
            iterationWeights=[frp_it_weight1,frp_it_weight2,frp_it_weight3],
            nodeSelfInfluence=node_self_infl,
            relationshipWeightProperty=rel_weight_prop,
            randomSeed=frp_seed,
            propertyRatio=prop_rat,
            featureProperties=feat_prop,
            writeProperty="emb_frp",            
        )

##### node2vec embedding creation

# Description of hyperparameters can be found: https://neo4j.com/docs/graph-data-science/current/algorithms/node2vec/
with st.expander("node2vec embedding creation"):
    n2v_seed = st.slider("Random seed:", value=42, min_value=1, max_value=99)
    n2v_dim = st.select_slider("Embedding dimenson", value=256, options=[128, 256, 512, 1024], key="n2v_dim")
    n2v_walk_length = st.slider("Walk length", value=80, min_value=2, max_value=160)
    n2v_walks_node = st.slider("Walks per node", value=10, min_value=2, max_value=50)
    n2v_io_factor = st.slider("inOutFactor", value=1.0, min_value=0.001, max_value=1.0, step=0.05)
    n2v_ret_factor = st.slider("returnFactor", value=1.0, min_value=0.001, max_value=1.0, step=0.05)
    n2v_neg_samp_rate = st.slider("negativeSamplingRate", value=5, min_value=5, max_value=20)
    n2v_iterations = st.slider("Number of training iterations", value=1, min_value=1, max_value=10)
    n2v_init_lr = st.select_slider("Initial learning rate", value=0.01, options=[0.001, 0.005, 0.01, 0.05, 0.1])
    n2v_min_lr = st.select_slider("Minimum learning rate", value=0.0001, options=[0.0001, 0.0005, 0.001, 0.005, 0.01])
    n2v_walk_bs = st.slider("Walk buffer size", value=1000, min_value=100, max_value=2000)
    rel_weight_prop = st.selectbox("Relationship weight property", rprop, key="n2v_rwp")

    if st.button("Create node2vec embedding"):
        st.session_state["gds"].node2vec.write(
            G,
            embeddingDimension=n2v_dim,
            walkLength=n2v_walk_length,
            walksPerNode=n2v_walks_node,
            inOutFactor=n2v_io_factor,
            returnFactor=n2v_ret_factor,
            negativeSamplingRate=n2v_neg_samp_rate,
            iterations=n2v_iterations,
            initialLearningRate=n2v_init_lr,
            minLearningRate=n2v_min_lr,
            walkBufferSize=n2v_walk_bs,
            relationshipWeightProperty=rel_weight_prop,
            randomSeed=n2v_seed,
            writeProperty="emb_n2v",           
        )

query = """
MATCH (n)
RETURN n.name AS name, n.emb_frp AS emb_frp, n.emb_n2v AS emb_n2v
"""
result = st.session_state["gds"].run_cypher(query)

st.write(result)

# if st.button("Drop embeddings"):
#     query = """
#     MATCH (n) REMOVE n.emb_frp
#     """
#     cypher.run(query)  
#     query = """
#     MATCH (n) REMOVE n.emb_n2v
#     """
#     cypher.run(query)     


#####
#
# t-SNE
#
#####

st.divider()
st.header("t-SNE")

emb = st.selectbox("Choose an embedding to plot: ", ["emb_frp", "emb_n2v"])

if st.button("Plot embeddings"):
    if st.session_state['data'] == "euro_roads":
        query = f"""
        MATCH (p:Place)-[:IN_COUNTRY]->(country)
        WHERE country.code IN ["E", "GB", "F", "TR", "I", "D", "GR"]
        RETURN p.name AS name, p.{emb} AS emb, country.code AS category
        """
    else:
        st.error("No embedding data is loaded!")
        st.stop()

    result = cypher.run(query)

    X = np.array(list(result["emb"]))
    X_embedded = TSNE(n_components=2, random_state=6).fit_transform(X)

    names = result["name"]
    categories = result["category"]
    tsne_df = pd.DataFrame(data = {
        "name": names,
        "category": categories,
        "x": [value[0] for value in X_embedded],
        "y": [value[1] for value in X_embedded],
    })

    chart = alt.Chart(tsne_df).mark_circle(size=60).encode(
    x="x",
    y="y",
    color="category",
    tooltip=["name", "category"]
    ).properties(width=700, height=400)

    st.altair_chart(chart, use_container_width=True)

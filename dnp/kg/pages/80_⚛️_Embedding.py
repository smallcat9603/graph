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
   st.title(f"{st.session_state['data']} Dataset Embedding Visualizer")

st.divider()
st.header("In-memory graph list")

graph_ls = st.session_state["gds"].graph.list()["graphName"]
if len(graph_ls) > 0:
    for el in graph_ls:
        st.info(el)
else:
    st.info('There are currently no graphs in memory.')

##### Create in-memory graphs
        
st.divider()
st.header("Create in-memory graph")

nodes = cypher.get_node_labels()
relationships = cypher.get_relationship_types()

node_label_list = st.multiselect("Node Labels", nodes, nodes)
relationship_type = st.selectbox("Relationship Type", relationships)
relationship_properties = cypher.get_relationship_properties(relationship_type)
relationship_property_list = st.multiselect("Relationship Properties", relationship_properties, relationship_properties)

if st.button("Create in-memory graph"):
    G, result = flow.project_graph(node_label_list, relationship_type, relationship_property_list)
    st.info(f"Graph {st.session_state['graph_name']} has {G.node_count()} nodes and {G.relationship_count()} relationships.")

##### Drop in-memory graph
    
st.divider()
st.header("Drop in-memory graph")

drop_graph = st.selectbox('Choose an graph to drop: ', st.session_state["gds"].graph.list()["graphName"])
if st.button('Drop in-memory graph'):
    flow.drop_memory_graph(drop_graph)
    st.info(f"Graph {drop_graph} has been dropped.")

##############################
#
#   Main panel content
#
##############################

def create_graph_df():
    df_query = """MATCH (n) RETURN n.name, n.frp_emb, n.n2v_emb"""
    df = pd.DataFrame([dict(_) for _ in cypher.run(df_query)])

    return df

def create_tsne_plot(emb_name='p.n2v_emb', n_components=2):
    tsne_query = """MATCH (p:Person) RETURN p.name AS name, p.death_year AS death_year, {} AS vec
    """.format(emb_name)
    df = pd.DataFrame([dict(_) for _ in cypher.run(tsne_query)])
    df['is_dead'] = np.where(df['death_year'].isnull(), 1, 0)

    X_emb = TSNE(n_components=n_components).fit_transform(list(df['vec']))

    tsne_df = pd.DataFrame(data = {
        'x': [value[0] for value in X_emb],
        'y': [value[1] for value in X_emb], 
        'label': df['is_dead']
    })

    return tsne_df

##############################

col1, col2 = st.columns((1, 2))

#####
#
# Embedding column (col1)
#
#####

with col1:
    emb_graph = st.selectbox('Enter graph name for embedding creation: ', st.session_state["gds"].graph.list()["graphName"])

##### FastRP embedding creation

    with st.expander('FastRP embedding creation'):
        st.markdown("Description of hyperparameters can be found [here](https://neo4j.com/docs/graph-data-science/current/algorithms/fastrp/#algorithms-embeddings-fastrp)")
        frp_dim = st.slider('FastRP embedding dimenson', value=4, min_value=2, max_value=50)
        frp_it_weight1 = st.slider('Iteration weight 1', value=0., min_value=0., max_value=1.)
        frp_it_weight2 = st.slider('Iteration weight 2', value=1., min_value=0., max_value=1.)
        frp_it_weight3 = st.slider('Iteration weight 3', value=1., min_value=0., max_value=1.)
        frp_norm = st.slider('FRP normalization strength', value=0., min_value=-1., max_value=1.)
        frp_seed = st.slider('Random seed', value=42, min_value=1, max_value=99)

        if st.button('Create FastRP embedding'):
            frp_query = f"""CALL gds.fastRP.write('{emb_graph}', {
                            embeddingDimension: {frp_dim},
                            iterationWeights: [{frp_it_weight1}, {frp_it_weight1}, {frp_it_weight1}],
                            normalizationStrength: {frp_norm},
                            randomSeed: {frp_seed},
                            writeProperty: 'frp_emb'
            })
            """
            result = cypher.run(frp_query)

##### node2vec embedding creation

    with st.expander('node2vec embedding creation'):
        st.markdown("Description of hyperparameters can be found [here](https://neo4j.com/docs/graph-data-science/current/algorithms/node2vec/)")
        n2v_dim = st.slider('node2vec embedding dimenson', value=4, min_value=2, max_value=50)
        n2v_walk_length = st.slider('Walk length', value=80, min_value=2, max_value=160)
        n2v_walks_node = st.slider('Walks per node', value=10, min_value=2, max_value=50)
        n2v_io_factor = st.slider('inOutFactor', value=1.0, min_value=0.001, max_value=1.0, step=0.05)
        n2v_ret_factor = st.slider('returnFactor', value=1.0, min_value=0.001, max_value=1.0, step=0.05)
        n2v_neg_samp_rate = st.slider('negativeSamplingRate', value=10, min_value=5, max_value=20)
        n2v_iterations = st.slider('Number of training iterations', value=1, min_value=1, max_value=10)
        n2v_init_lr = st.select_slider('Initial learning rate', value=0.01, options=[0.001, 0.005, 0.01, 0.05, 0.1])
        n2v_min_lr = st.select_slider('Minimum learning rate', value=0.0001, options=[0.0001, 0.0005, 0.001, 0.005, 0.01])
        n2v_walk_bs = st.slider('Walk buffer size', value=1000, min_value=100, max_value=2000)
        n2v_seed = st.slider('Random seed:', value=42, min_value=1, max_value=99)

        if st.button('Create node2vec embedding'):
            n2v_query = f"""CALL gds.node2vec.write('{emb_graph}', {
                            embeddingDimension: {n2v_dim},
                            walkLength: {n2v_walk_length},
                            walksPerNode: {n2v_walks_node},
                            inOutFactor: {n2v_io_factor},
                            returnFactor: {n2v_ret_factor},
                            negativeSamplingRate: {n2v_neg_samp_rate},
                            iterations: {n2v_iterations},
                            initialLearningRate: {n2v_init_lr},
                            minLearningRate: {n2v_min_lr},
                            walkBufferSize: {n2v_walk_bs},
                            randomSeed: {n2v_seed},
                            writeProperty: 'n2v_emb'
            })
            """
            result = cypher.run(n2v_query)

    st.markdown("---")

    if st.button('Show embeddings'):
        df = create_graph_df()
        st.dataframe(df)

    if st.button('Drop embeddings'):
        cypher.run('MATCH (n) REMOVE n.frp_emb')
        cypher.run('MATCH (n) REMOVE n.n2v_emb')

#####
#
# t-SNE column (col2)
#
#####

with col2:
    st.header('t-SNE')

    plt_emb = st.selectbox('Choose an embedding to plot: ', ['FastRP', 'node2vec'])
    if plt_emb == 'FastRP':
        emb_name = 'p.frp_emb'
    else:
        emb_name = 'p.n2v_emb'

    if st.button('Plot embeddings'):

        tsne_df = create_tsne_plot(emb_name=emb_name)
        ch_alt = alt.Chart(tsne_df).mark_point().encode(
            x='x', 
            y='y', 
            color=alt.Color('label:O', scale=alt.Scale(range=['red', 'blue']))
        ).properties(width=800, height=800)
        st.altair_chart(ch_alt, use_container_width=True)
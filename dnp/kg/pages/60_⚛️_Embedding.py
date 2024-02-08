import streamlit as st
from pages.lib import cypher, flow

if 'data' not in st.session_state:
   st.title("No Graph Data")
   st.warning("You should load graph data first!", icon='⚠')
   st.stop()
else:
   st.title(f"In-memory Graph Embedding ({st.session_state['data']})")

##### List in-memory graph

graph_ls = st.session_state["gds"].graph.list()["graphName"]
if len(graph_ls) > 0:
    for g in graph_ls:
        G = st.session_state["gds"].graph.get(g)
        st.success(f"Graph {g}: {G.node_count()} nodes and {G.relationship_count()} relationships")  
else:
    st.warning('There are currently no graphs in memory.')
    st.stop()

##############################
#
#   Embedding
#
##############################

emb_graph = st.selectbox('Enter graph name for embedding creation: ', graph_ls)
G = st.session_state["gds"].graph.get(emb_graph)

rprop = [None]
for rt in G.relationship_types():
    rprop += G.relationship_properties(rt)
nprop = []
for nl in G.node_labels():
    nprop += G.node_properties(nl)

embs = ["emb_frp", "emb_n2v"]

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
        flow.node_emb_frp(G, 
            embeddingDimension=frp_dim,
            normalizationStrength=frp_norm,
            iterationWeights=[frp_it_weight1,frp_it_weight2,frp_it_weight3],
            nodeSelfInfluence=node_self_infl,
            relationshipWeightProperty=rel_weight_prop,
            randomSeed=frp_seed,
            propertyRatio=prop_rat,
            featureProperties=feat_prop
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
        flow.node_emb_n2v(
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
            randomSeed=n2v_seed          
        )

with st.expander("Show & Drop embeddings"):
    if st.button("Show embeddings"):
        result = cypher.update_emb_result()
        st.write(result)

    if st.button("Drop embeddings"):
        cypher.drop_emb()    


#####
#
# t-SNE
#
#####

st.divider()
st.header("t-SNE")

form = st.form("t-SNE")
emb = form.radio("Choose an embedding to plot:", 
                    embs, 
                    captions=["FastRP", "node2vec"],
                    horizontal=True)

if form.form_submit_button("Plot embeddings"):
    if cypher.update_emb_status(emb) == False:
        st.warning("You should do embedding first!")
        st.stop()

    result = cypher.get_emb_result(emb)

    flow.plot_tsne_alt(result)

#####
#
# kNN
#
#####  

st.divider()
st.header("kNN") 

if "query" not in st.session_state:
    st.warning(f"No query node in dataset {st.session_state['data']}!")
    st.stop()

embs_done = []
for emb in embs:
    if cypher.update_emb_status(emb):
        embs_done.append(emb)

if len(embs_done) == 0:
    st.warning("You should do embedding first!")
else:
    for emb in embs_done:
        if st.session_state["load"] != "Offline": 
            result = flow.kNN(G, emb)
            with st.expander("Debug Info"):
                st.write(f"Relationships produced: {result['relationshipsWritten']}")
                st.write(f"Nodes compared: {result['nodesCompared']}")
                st.write(f"Mean similarity: {result['similarityDistribution']['mean']}")

        st.subheader(emb)
        result = cypher.interact_knn(emb, st.session_state["query"]) # evaluate (node embedding + knn)
        with st.expander("Debug Info"):
            st.code(result)

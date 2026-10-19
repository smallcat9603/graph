import numpy as np
import streamlit as st

from embkg import evaluate, graphio, paths, tuning
from ui import core, views


st.title("RW-based Embedding")
st.divider()

graph_tool = st.radio("Select one graph tool:",
                ["igraph", "networkx"],
                horizontal=True,
                )
edgefile = st.file_uploader("Choose an edge file:")
labelfile = st.file_uploader("Choose a label file:")
labelfile2 = st.file_uploader("Choose a second label file:")
df_label = graphio.read_labels(labelfile)
df_label2 = graphio.read_labels(labelfile2)
if edgefile is not None:
    df = graphio.read_edges(edgefile)

    st.divider()
    st.header("Graph Info")
    try:
        G = graphio.build_graph(df, graph_tool)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.info(G.summary() if graph_tool == "igraph" else G)

    form = st.form("emb")

    sim = form.radio("Select the similarity metric:",
                    ["Autocovariance", "PMI"],
                    horizontal=True,
                    )
    tau = form.slider("Markov time:",
                    1,
                    100,
                    3)
    dim = form.select_slider("Dimension:",
                             value=128,
                             options=[128, 256, 512, 1024])
    alpha = form.select_slider("Topology importance (topo-to-label):",
                                value=1.0,
                                options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    beta = form.select_slider("Label importance (label1-to-label2):",
                                value=1.0,
                                options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    save_emb = form.checkbox("Save embedding result")

    if form.form_submit_button("Embedding"):
        nrows = 10

        def show_step(title, mat):
            st.header(f"{title} ({nrows} rows)")
            st.write(mat.shape)
            st.table(mat[:nrows, :])

        name = edgefile.name[:edgefile.name.rfind('.')]
        u, category = core.node_emb(G, name, sim, tau, dim, graph_tool, df_label, df_label2, alpha, beta, _on_step=show_step)
        emb_df = evaluate.emb_frame(u, category)

        if save_emb:
            paths.EMB_OUT_DIR.mkdir(parents=True, exist_ok=True)
            out = paths.EMB_OUT_DIR / f"{name}_tau{tau}_dim{dim}_alpha{alpha}_beta{beta}.txt"
            np.savetxt(out, u)
            st.success(f"Saved to {out}")

        st.header("t-SNE")
        views.plot_tsne_alt(emb_df)

        st.header("ML")
        if len(np.unique(emb_df["category"])) > 1:
            views.modeler(emb_df)
        else:
            st.warning(f"The dataset has only one category!")

    if form.form_submit_button("Tune hyperparameters"):
        initial_params = {
            "sim": sim,
            "tau": tau,
            "dim": dim,
            "alpha": alpha,
            "beta": beta,
        }
        views.tune(tuning.objective_emb(G, graph_tool, df_label, df_label2), initial_params)

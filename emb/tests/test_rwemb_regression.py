"""embkg.rwemb must reproduce dnp/kg (the code behind the published RWEM results).

- Deterministic stages (transition matrix, label mixing, autocovariance, PMI) are
  compared exactly with the original functions in dnp/kg/pages/lib/flow.py.
- The topology-only embedding (alpha=1) is compared, to 1e-5, with the file the
  original app saved (dnp/kg/emb/airport_0_tau3_dim128_alpha1.0_beta1.0.txt).
- With labels the final eigsh step is not reproducible even run to run
  (measurement_log #2), so it is not compared bit for bit.
"""

import importlib.util
import sys

import numpy as np
import pytest

from embkg import graphio, paths, rwemb

AIRPORT_EDGES = paths.GRAPH_DATA_DIR / "airport_0.edges"
CONTINENT = paths.GRAPH_DATA_DIR / "airport.continent_0.labels"
COUNTRY = paths.GRAPH_DATA_DIR / "airport.country_0.labels"
SAVED_TOPO = paths.KG_DIR / "emb" / "airport_0_tau3_dim128_alpha1.0_beta1.0.txt"


@pytest.fixture(scope="module")
def flow():
    """Import the original dnp/kg flow module without modifying dnp/kg."""
    sys.path.insert(0, str(paths.KG_DIR))
    sys.dont_write_bytecode = True # no __pycache__ inside dnp/kg
    try:
        spec = importlib.util.spec_from_file_location("kg_flow", paths.KG_DIR / "pages" / "lib" / "flow.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(paths.KG_DIR))


@pytest.fixture(scope="module", params=["igraph", "networkx"])
def graph(request):
    return request.param, graphio.build_graph(graphio.read_edges(AIRPORT_EDGES), request.param)


def test_transition_matrix(flow, graph):
    tool, G = graph
    np.testing.assert_array_equal(
        rwemb.standard_random_walk_transition_matrix(G, tool),
        flow.standard_random_walk_transition_matrix(G, tool))


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.5), (0.5, 0.1), (0.8, 1.0)])
def test_label_mixing_and_similarity(flow, graph, alpha, beta):
    tool, G = graph
    M = rwemb.standard_random_walk_transition_matrix(G, tool)
    l1 = rwemb.get_node_labels(graphio.read_labels(CONTINENT))
    l2 = rwemb.get_node_labels(graphio.read_labels(COUNTRY))
    n = M.shape[0]

    ours = rwemb.M_attr(n, M.copy(), l1, l2, alpha, beta)
    theirs = flow.M_attr(n=n, M=M.copy(), node_labels=l1, node_labels2=l2, alpha=alpha, beta=beta)
    np.testing.assert_array_equal(ours, theirs)
    assert rwemb.get_category_list(l1) == flow.get_category_list(l1)

    np.testing.assert_array_equal(rwemb.autocovariance_matrix(ours, 3), flow.autocovariance_matrix(theirs, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_array_equal(
            rwemb.preprocess_similarity_matrix(rwemb.PMI_matrix(ours, 3)),
            flow.preprocess_similarity_matrix(flow.PMI_matrix(theirs, 3)))


def test_topology_embedding_matches_saved_file():
    G = graphio.build_graph(graphio.read_edges(AIRPORT_EDGES), "igraph")
    u, category = rwemb.node_emb(G, "Autocovariance", 3, 128, "igraph",
                                 df_label=graphio.read_labels(CONTINENT), alpha=1.0, beta=1.0)
    # eigsh starts from a random vector, so repeated runs differ by ~1e-6 here.
    np.testing.assert_allclose(u, np.loadtxt(SAVED_TOPO), rtol=0, atol=1e-5)
    assert len(category) == u.shape[0]

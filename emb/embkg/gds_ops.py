"""Neo4j Graph Data Science calls: node similarity, PPR, FastRP, node2vec, kNN."""


def drop_memory_graph(gds, graph_name):
    if gds.graph.exists(graph_name)["exists"]:
        gds.graph.get(graph_name).drop()


def write_nodesimilarity(gds, G, metric):
    """Filtered node similarity Query -> Article, written as SIMILAR_<metric>."""
    return gds.nodeSimilarity.filtered.write(
        G,
        similarityMetric=metric,
        writeRelationshipType=f'SIMILAR_{metric}',
        writeProperty='score',
        relationshipWeightProperty="weight",
        sourceNodeFilter="Query",
        targetNodeFilter="Article",
        topK=100,
    )


def write_nodesimilarity_ppr(gds, G, QUERY_DICT):
    """Personalized PageRank from each query node, written as pr0, pr1, ..."""
    results = ""
    for idx, name in enumerate(list(QUERY_DICT.keys())):
        nodeid = gds.find_node_id(labels=["Query"], properties={"name": name})
        result = gds.pageRank.write(
            G,
            writeProperty="pr"+str(idx),
            maxIterations=20,
            dampingFactor=0.85,
            relationshipWeightProperty='weight',
            sourceNodes=[nodeid]
        )
        results += f"Node properties written: {result['nodePropertiesWritten']}\n"
        results += f"Mean: {result['centralityDistribution']['mean']}\n"
    return results


def node_emb_frp(gds, G, downstream=True, **params):
    """FastRP: mutate (for downstream kNN) and write (for t-SNE) as emb_frp."""
    if downstream:
        gds.fastRP.mutate(G, mutateProperty="emb_frp", **params)
    gds.fastRP.write(G, writeProperty="emb_frp", **params)


def node_emb_n2v(gds, G, downstream=True, **params):
    """node2vec: mutate (for downstream kNN) and write (for t-SNE) as emb_n2v."""
    if downstream:
        gds.node2vec.mutate(G, mutateProperty="emb_n2v", **params)
    gds.node2vec.write(G, writeProperty="emb_n2v", **params)


KNN_REL_TYPE = {
    "emb_frp": "SIMILAR_FASTRP",
    "emb_n2v": "SIMILAR_NODE2VEC",
    "emb_hgn": "SIMILAR_HASHGNN",
}


def knn(gds, G, emb, topK=100, writeProperty="score", sourceNodeFilter="Query", targetNodeFilter="Article"):
    """Filtered kNN on the in-memory (mutated) embedding property `emb`."""
    return gds.knn.filtered.write(
        G,
        topK=topK,
        nodeProperties=[emb],
        randomSeed=42, # requires concurrency=1
        concurrency=1,
        sampleRate=1.0,
        deltaThreshold=0.0,
        writeRelationshipType=KNN_REL_TYPE[emb],
        writeProperty=writeProperty,
        sourceNodeFilter=sourceNodeFilter,
        targetNodeFilter=targetNodeFilter,
    )

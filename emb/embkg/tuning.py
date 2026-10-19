"""Optuna objectives: maximize mean SVM accuracy of an embedding."""

import numpy as np

from embkg import evaluate, gds_ops, rwemb


def mean_accuracy(result, seed=None):
    return np.mean(evaluate.svm_accuracy(result, seed=seed)[0])


def objective_frp(gds, G, rel_weight_prop, prop_rat, feat_prop, fetch_result):
    """fetch_result(emb_property) -> DataFrame (name, emb, category) read back from Neo4j."""
    def objective(trial):
        params = {
            "embeddingDimension": trial.suggest_int("embeddingDimension", 128, 512, log=True),
            "normalizationStrength": trial.suggest_float("normalizationStrength", -1.0, 1.0),
            "iterationWeights1": trial.suggest_float("iterationWeights1", 0.0, 1.0),
            "iterationWeights2": trial.suggest_float("iterationWeights2", 0.0, 1.0),
            "iterationWeights3": trial.suggest_float("iterationWeights3", 0.0, 1.0),
            "nodeSelfInfluence": trial.suggest_float("nodeSelfInfluence", 0.0, 1.0),
            "randomSeed": trial.suggest_int("randomSeed", 1, 99),
        }

        gds_ops.node_emb_frp(
            gds, G,
            embeddingDimension=params["embeddingDimension"],
            normalizationStrength=params["normalizationStrength"],
            iterationWeights=[params["iterationWeights1"], params["iterationWeights2"], params["iterationWeights3"]],
            nodeSelfInfluence=params["nodeSelfInfluence"],
            relationshipWeightProperty=rel_weight_prop,
            randomSeed=params["randomSeed"],
            propertyRatio=prop_rat,
            featureProperties=feat_prop,
            downstream=False,
        )

        return mean_accuracy(fetch_result("emb_frp"))

    return objective


def objective_n2v(gds, G, rel_weight_prop, fetch_result):
    """fetch_result(emb_property) -> DataFrame (name, emb, category) read back from Neo4j."""
    def objective(trial):
        params = {
            "embeddingDimension": trial.suggest_int("embeddingDimension", 128, 512, log=True),
            "walkLength": trial.suggest_int("walkLength", 2, 160),
            "walksPerNode": trial.suggest_int("walksPerNode", 2, 50),
            "inOutFactor": trial.suggest_float("inOutFactor", 0.001, 1.0, step=0.05),
            "returnFactor": trial.suggest_float("returnFactor", 0.001, 1.0, step=0.05),
            "negativeSamplingRate": trial.suggest_int("negativeSamplingRate", 5, 20),
            "iterations": trial.suggest_int("iterations", 1, 10),
            "initialLearningRate": trial.suggest_categorical("initialLearningRate", [0.001, 0.005, 0.01, 0.05, 0.1]),
            "minLearningRate": trial.suggest_categorical("minLearningRate", [0.0001, 0.0005, 0.001, 0.005, 0.01]),
            "walkBufferSize": trial.suggest_int("walkBufferSize", 100, 2000),
            "randomSeed": trial.suggest_int("randomSeed", 1, 99),
        }

        gds_ops.node_emb_n2v(gds, G, relationshipWeightProperty=rel_weight_prop, downstream=False, **params)

        return mean_accuracy(fetch_result("emb_n2v"))

    return objective


def objective_emb(G, graph_tool, df_label, df_label2, seed=None):
    """RWEM objective; needs no database."""
    def objective(trial):
        params = {
            "sim": trial.suggest_categorical("sim", ["Autocovariance", "PMI"]),
            "tau": trial.suggest_int("tau", 1, 100),
            "dim": trial.suggest_int("dim", 128, 1024, log=True),
            "alpha": trial.suggest_float("alpha", 0.1, 1.0, step=0.1),
            "beta": trial.suggest_float("beta", 0.1, 1.0, step=0.1),
        }

        u, category = rwemb.node_emb(
            G, graph_tool=graph_tool,
            df_label=None if df_label is None else df_label.copy(),
            df_label2=None if df_label2 is None else df_label2.copy(),
            **params,
        )

        return mean_accuracy(evaluate.emb_frame(u, category), seed=seed)

    return objective

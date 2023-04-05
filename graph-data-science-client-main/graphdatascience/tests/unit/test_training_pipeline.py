import pytest

from .conftest import CollectingQueryRunner
from graphdatascience.graph_data_science import GraphDataScience
from graphdatascience.pipeline.lp_training_pipeline import LPTrainingPipeline
from graphdatascience.pipeline.nc_training_pipeline import NCTrainingPipeline
from graphdatascience.pipeline.nr_training_pipeline import NRTrainingPipeline

PIPE_NAME = "pipe"


@pytest.fixture
def lp_pipe(gds: GraphDataScience) -> LPTrainingPipeline:
    pipe, _ = gds.beta.pipeline.linkPrediction.create(PIPE_NAME)
    return pipe


@pytest.fixture
def nc_pipe(gds: GraphDataScience) -> NCTrainingPipeline:
    pipe, _ = gds.beta.pipeline.nodeClassification.create(PIPE_NAME)
    return pipe


@pytest.fixture
def nr_pipe(gds: GraphDataScience) -> NRTrainingPipeline:
    pipe, _ = gds.alpha.pipeline.nodeRegression.create(PIPE_NAME)
    return pipe


def test_create_lp_pipeline(runner: CollectingQueryRunner, gds: GraphDataScience) -> None:
    lp_pipe, _ = gds.beta.pipeline.linkPrediction.create("hello")
    assert lp_pipe.name() == "hello"

    assert runner.last_query() == "CALL gds.beta.pipeline.linkPrediction.create($name)"
    assert runner.last_params() == {
        "name": "hello",
    }


def test_add_node_property_lp_pipeline(runner: CollectingQueryRunner, lp_pipe: LPTrainingPipeline) -> None:
    lp_pipe.addNodeProperty("pageRank", mutateProperty="rank", dampingFactor=0.2, tolerance=0.3)

    assert (
        runner.last_query()
        == "CALL gds.beta.pipeline.linkPrediction.addNodeProperty($pipeline_name, $procedure_name, $config)"
    )
    assert runner.last_params() == {
        "pipeline_name": lp_pipe.name(),
        "procedure_name": "pageRank",
        "config": {"mutateProperty": "rank", "dampingFactor": 0.2, "tolerance": 0.3},
    }


def test_add_feature_lp_pipeline(runner: CollectingQueryRunner, lp_pipe: LPTrainingPipeline) -> None:
    lp_pipe.addFeature("l2", nodeProperties=["prop1"])

    assert (
        runner.last_query()
        == "CALL gds.beta.pipeline.linkPrediction.addFeature($pipeline_name, $feature_type, $config)"
    )
    assert runner.last_params() == {
        "pipeline_name": lp_pipe.name(),
        "feature_type": "l2",
        "config": {"nodeProperties": ["prop1"]},
    }


def test_configure_split_lp_pipeline(runner: CollectingQueryRunner, lp_pipe: LPTrainingPipeline) -> None:
    lp_pipe.configureSplit(trainFraction=0.42)

    assert runner.last_query() == "CALL gds.beta.pipeline.linkPrediction.configureSplit($pipeline_name, $config)"
    assert runner.last_params() == {
        "pipeline_name": lp_pipe.name(),
        "config": {"trainFraction": 0.42},
    }


def test_add_logistic_regression_nc_pipeline(runner: CollectingQueryRunner, nc_pipe: NCTrainingPipeline) -> None:
    nc_pipe.addLogisticRegression(penalty=1)

    assert (
        runner.last_query()
        == "CALL gds.beta.pipeline.nodeClassification.addLogisticRegression($pipeline_name, $config)"
    )
    assert runner.last_params() == {
        "pipeline_name": nc_pipe.name(),
        "config": {"penalty": 1},
    }


def test_add_linear_regression_nr_pipeline(runner: CollectingQueryRunner, nr_pipe: NRTrainingPipeline) -> None:
    nr_pipe.addLinearRegression(penalty=1)

    assert runner.last_query() == "CALL gds.alpha.pipeline.nodeRegression.addLinearRegression($pipeline_name, $config)"
    assert runner.last_params() == {
        "pipeline_name": nr_pipe.name(),
        "config": {"penalty": 1},
    }


def test_add_random_forest_lp_pipeline(runner: CollectingQueryRunner, lp_pipe: LPTrainingPipeline) -> None:
    lp_pipe.addRandomForest(penalty=1)

    assert runner.last_query() == "CALL gds.alpha.pipeline.linkPrediction.addRandomForest($pipeline_name, $config)"
    assert runner.last_params() == {
        "pipeline_name": lp_pipe.name(),
        "config": {"penalty": 1},
    }


def test_add_random_forest_with_range_lp_pipeline(runner: CollectingQueryRunner, lp_pipe: LPTrainingPipeline) -> None:
    lp_pipe.addRandomForest(maxDepth=(1, 2))

    assert runner.last_query() == "CALL gds.alpha.pipeline.linkPrediction.addRandomForest($pipeline_name, $config)"
    assert runner.last_params() == {
        "pipeline_name": lp_pipe.name(),
        "config": {"maxDepth": {"range": [1, 2]}},
    }


def test_add_mlp_lp_pipeline(runner: CollectingQueryRunner, lp_pipe: LPTrainingPipeline) -> None:
    lp_pipe.addMLP(hiddenLayerSizes=[64, 16, 4], penalty=0.1)

    assert runner.last_query() == "CALL gds.alpha.pipeline.linkPrediction.addMLP($pipeline_name, $config)"
    assert runner.last_params() == {
        "pipeline_name": lp_pipe.name(),
        "config": {"hiddenLayerSizes": [64, 16, 4], "penalty": 0.1},
    }


def test_add_mlp_nc_pipeline(runner: CollectingQueryRunner, nc_pipe: NCTrainingPipeline) -> None:
    nc_pipe.addMLP(hiddenLayerSizes=[64, 16, 4], penalty=0.1)

    assert runner.last_query() == "CALL gds.alpha.pipeline.nodeClassification.addMLP($pipeline_name, $config)"
    assert runner.last_params() == {
        "pipeline_name": nc_pipe.name(),
        "config": {"hiddenLayerSizes": [64, 16, 4], "penalty": 0.1},
    }


def test_configure_auto_tuning_lp_pipeline(runner: CollectingQueryRunner, lp_pipe: LPTrainingPipeline) -> None:
    lp_pipe.configureAutoTuning(maxTrials=42)

    assert runner.last_query() == "CALL gds.alpha.pipeline.linkPrediction.configureAutoTuning($pipeline_name, $config)"
    assert runner.last_params() == {
        "pipeline_name": lp_pipe.name(),
        "config": {"maxTrials": 42},
    }


def test_train_estimate_lp_pipeline(
    runner: CollectingQueryRunner, gds: GraphDataScience, lp_pipe: LPTrainingPipeline
) -> None:
    G, _ = gds.graph.project("g", "*", "*")

    lp_pipe.train_estimate(G, modelName="m", concurrency=2)

    assert runner.last_query() == "CALL gds.beta.pipeline.linkPrediction.train.estimate($graph_name, $config)"
    assert runner.last_params() == {
        "graph_name": G.name(),
        "config": {"pipeline": lp_pipe.name(), "modelName": "m", "concurrency": 2},
    }


def test_train_lp_pipeline(runner: CollectingQueryRunner, gds: GraphDataScience, lp_pipe: LPTrainingPipeline) -> None:
    G, _ = gds.graph.project("g", "*", "*")

    lp_pipe.train(G, modelName="m", concurrency=2)

    assert runner.last_query() == "CALL gds.beta.pipeline.linkPrediction.train($graph_name, $config)"
    assert runner.last_params() == {
        "graph_name": G.name(),
        "config": {"pipeline": lp_pipe.name(), "modelName": "m", "concurrency": 2},
    }


def test_select_features_nc_pipeline(runner: CollectingQueryRunner, nc_pipe: NCTrainingPipeline) -> None:
    nc_pipe.selectFeatures("hello")

    assert (
        runner.last_query()
        == "CALL gds.beta.pipeline.nodeClassification.selectFeatures($pipeline_name, $node_properties)"
    )
    assert runner.last_params() == {
        "pipeline_name": nc_pipe.name(),
        "node_properties": "hello",
    }

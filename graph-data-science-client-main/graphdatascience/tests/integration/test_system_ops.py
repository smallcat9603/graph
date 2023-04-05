import pytest

from graphdatascience.graph_data_science import GraphDataScience


def test_listProgress(gds: GraphDataScience) -> None:
    result = gds.beta.listProgress()

    assert len(result) >= 0


def test_userLog(gds: GraphDataScience) -> None:
    result = gds.alpha.userLog()

    assert len(result) >= 0


@pytest.mark.enterprise
def test_systemMonitor(gds: GraphDataScience) -> None:
    result = gds.alpha.systemMonitor()

    assert result["freeHeap"] >= 0
    assert len(result["ongoingGdsProcedures"]) >= 0


@pytest.mark.skip_on_aura
def test_sysInfo(gds: GraphDataScience) -> None:
    result = gds.debug.sysInfo()

    assert "gdsVersion" in (list(result["key"]))


@pytest.mark.enterprise
def test_is_licensed(gds: GraphDataScience) -> None:
    assert gds.is_licensed()


@pytest.mark.skip_on_aura
def test_set_defaults(gds: GraphDataScience) -> None:
    gds.alpha.config.defaults.set("option1", 2, "")
    assert True


@pytest.mark.skip_on_aura
def test_list_defaults(gds: GraphDataScience) -> None:
    gds.alpha.config.defaults.set("option1", 2, "")
    gds.alpha.config.defaults.set("option2", 2, "")
    result = gds.alpha.config.defaults.list(username="")

    assert len(result) == 2


@pytest.mark.enterprise
@pytest.mark.skip_on_aura
def test_backup(gds: GraphDataScience) -> None:
    result = gds.alpha.backup(concurrency=4)

    assert len(result) == 0


@pytest.mark.enterprise
@pytest.mark.skip_on_aura
def test_restore(gds: GraphDataScience) -> None:
    result = gds.alpha.restore(concurrency=4)

    assert len(result) == 0

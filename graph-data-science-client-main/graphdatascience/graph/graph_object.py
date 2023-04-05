from types import TracebackType
from typing import Any, List, Optional, Type, TypeVar, Union

from pandas import Series

from ..query_runner.query_runner import QueryRunner
from ..server_version.server_version import ServerVersion

TGraph = TypeVar("TGraph", bound="Graph")


class Graph:
    def __init__(self, name: str, query_runner: QueryRunner, server_version: ServerVersion):
        self._name = name
        self._query_runner = query_runner
        self._server_version = server_version

    def __enter__(self: TGraph) -> TGraph:
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.drop()

    def name(self) -> str:
        return self._name

    def _graph_info(self, yields: List[str] = []) -> "Series[Any]":
        yield_suffix = "" if len(yields) == 0 else " YIELD " + ", ".join(yields)
        info = self._query_runner.run_query(
            f"CALL gds.graph.list($graph_name){yield_suffix}", {"graph_name": self._name}, custom_error=False
        )

        if len(info) == 0:
            raise ValueError(f"There is no projected graph named '{self.name()}'")

        return info.squeeze()  # type: ignore

    def database(self) -> str:
        return self._graph_info(["database"])  # type: ignore

    def configuration(self) -> "Series[Any]":
        return Series(self._graph_info(["configuration"]))

    def node_count(self) -> int:
        return self._graph_info(["nodeCount"])  # type: ignore

    def relationship_count(self) -> int:
        return self._graph_info(["relationshipCount"])  # type: ignore

    def node_labels(self) -> List[str]:
        return list(self._graph_info(["schema"])["nodes"].keys())

    def relationship_types(self) -> List[str]:
        return list(self._graph_info(["schema"])["relationships"].keys())

    def node_properties(self, label: Optional[str] = None) -> Union["Series[str]", List[str]]:
        labels_to_props = self._graph_info(["schema"])["nodes"]

        if not label:
            return Series({key: list(val.keys()) for key, val in labels_to_props.items()})

        if label not in labels_to_props.keys():
            raise ValueError(f"There is no node label '{label}' projected onto '{self.name()}'")

        return list(labels_to_props[label].keys())

    def relationship_properties(self, type: Optional[str] = None) -> Union["Series[str]", List[str]]:
        types_to_props = self._graph_info(["schema"])["relationships"]

        if not type:
            return Series({key: list(val.keys()) for key, val in types_to_props.items()})

        if type not in types_to_props.keys():
            raise ValueError(f"There is no relationship type '{type}' projected onto '{self.name()}'")

        return list(types_to_props[type].keys())

    def degree_distribution(self) -> "Series[float]":
        return Series(self._graph_info(["degreeDistribution"]))

    def density(self) -> float:
        return self._graph_info(["density"])  # type: ignore

    def memory_usage(self) -> str:
        return self._graph_info(["memoryUsage"])  # type: ignore

    def size_in_bytes(self) -> int:
        return self._graph_info(["sizeInBytes"])  # type: ignore

    def exists(self) -> bool:
        result = self._query_runner.run_query(
            "CALL gds.graph.exists($graph_name)",
            {"graph_name": self._name},
            custom_error=False,
        )
        return result.squeeze()["exists"]  # type: ignore

    def drop(self, failIfMissing: bool = False) -> "Series[str]":
        result = self._query_runner.run_query(
            "CALL gds.graph.drop($graph_name, $fail_if_missing)",
            {"graph_name": self._name, "fail_if_missing": failIfMissing},
            custom_error=False,
        )

        return result.squeeze()  # type: ignore

    def creation_time(self) -> Any:  # neo4j.time.DateTime not exported
        return self._graph_info(["creationTime"])

    def modification_time(self) -> Any:  # neo4j.time.DateTime not exported
        return self._graph_info(["modificationTime"])

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name()}, "
            f"node_count={self.node_count()}, relationship_count={self.relationship_count()})"
        )

    def __repr__(self) -> str:
        yield_fields = [
            "graphName",
            "nodeCount",
            "relationshipCount",
            "database",
            "configuration",
            "schema",
            "memoryUsage",
        ]
        return f"{self.__class__.__name__}({self._graph_info(yields=yield_fields).to_dict()})"

"""Recommend additional features to add to a dataset."""
from typing import Dict, List, Tuple

import pandas as pd

from graphtester.evaluate.dataset import (
    DEFAULT_METRICS,
    EvaluationResult,
    _clean_graphs,
    _Metric,
    evaluate,
)
from graphtester.io.dataset import Dataset
from graphtester.label import EDGE_LABELING_METHODS, VERTEX_LABELING_METHODS


class RecommendationResult:
    """The result of a recommendation."""

    def __init__(
        self,
        dataset: Dataset,
        results: Dict[Tuple[str, Tuple[str]], EvaluationResult],
        metrics: List[_Metric],
        iterations: int,
    ):
        """Initialize a recommendation result.

        Parameters
        ----------
        dataset : Dataset
            The dataset that was recommended features for.
        results : Dict[ Tuple[str, Tuple[str]], EvaluationResult]
            The evaluation results for each recommended feature. The index
            of this dictionary is a tuple of the state of the dataset (with or
            without node or edge features) and the features that were added to
            achieve the values at the EvaluationResult.
        metrics : List[_Metric]
            The metrics that were evaluated.
        iterations : int
            The number of iterations that were performed.
        """
        self.dataset_name = dataset.name
        self.results = results
        self.metrics = metrics
        self.iterations = iterations
        self._dataframe = None

    def __repr__(self) -> str:
        """Return the representation of a recommendation result."""
        return f"RecommendationResult(dataset={self.dataset_name})"

    def as_dataframe(self) -> pd.DataFrame:  # noqa: C901
        """Return the recommendation result as a pandas dataframe."""
        if self._dataframe is not None:
            return self._dataframe

        rows_dict = {}
        # we will create different datasets for each state
        # before merging them into a single dataframe
        for (state, features), results in self.results.items():
            if state not in rows_dict:
                rows_dict[state] = []
            data = {
                "Feature count": len(features),
                "Feature name(s)": " + ".join(
                    [
                        f"{f} ({'n' if f in VERTEX_LABELING_METHODS else 'e'})"
                        for f in features
                    ]
                )
                if features
                else "-",
            }
            for metric in self.metrics:
                data[metric.description] = results.results[metric.name][self.iterations]
            rows_dict[state].append(data)

        dataframes = []
        for idx, (state, rows) in enumerate(rows_dict.items()):
            df = pd.DataFrame(rows)
            # Drop features that are not in top 5 per feature count and state
            sort_by = [("Feature count", True)]
            for metric in self.metrics:
                sort_by.append((metric.description, not metric.higher_is_better))
            sort_by.append(("Feature name(s)", False))
            df = df.sort_values(
                by=[by for by, _ in sort_by], ascending=[asc for _, asc in sort_by]
            )
            df = df.groupby("Feature count").head(5).reset_index(drop=True)
            if idx != 0:
                df = df.drop(columns=["Feature count"])
            dataframes.append(df)

        states = list(rows_dict.keys())
        report = pd.concat(dataframes, axis=1, keys=states)
        report = report.set_index((states[0], "Feature count"))
        report.index.name = "Feature count"
        report.name = self.dataset_name
        report = report.round(4)
        self._dataframe = report
        return report

    def __str__(self) -> str:
        """Return the string representation of a recommendation result."""
        return self.as_dataframe().to_string()


def recommend(
    dataset: Dataset,
    metrics: List[str],
    max_feature_count=3,
    features_to_test=None,
    node_features=True,
    edge_features=True,
    ignore_original_features=False,
    fast=True,
    iterations=1,
) -> RecommendationResult:
    """Recommend additional features (methods) to add to a dataset.

    Test all possible features and recommend the ones that add the most
    1-WL-efficiency to the dataset. Features are tested by adding them to the
    dataset one at a time and evaluating the dataset. The feature that adds the
    most efficiency is then added to the dataset and the process is repeated until
    all features have been tested, or full efficiency has been reached in all
    dimensions.

    Parameters
    ----------
    dataset : Dataset
        The dataset to recommend features for.
    metrics : List[str]
        The metrics to evaluate the dataset on.
    max_feature_count : int, optional
        The maximum number of features to combine into a set, by default 3
    features_to_test : List[str], optional
        The features to test, by default None. If None, all features will be tested,
        depending on node_features, edge_features and fast arguments (see below).
    node_features : bool, optional
        Whether to recommend node features, by default True
    edge_features : bool, optional
        Whether to recommend edge features, by default True
    ignore_original_features : bool, optional
        Whether to ignore the original features of the dataset, by default False
    fast : bool, optional
        Whether to only use the features that are scalable to large datasets,
        by default True. Ignored if features_to_test is not None.
    iterations : int, optional
        The number of iterations to run the comparison for, by default 1

    Returns
    -------
    RecommendationResult
        The recommendation result.
    """
    if not isinstance(metrics[0], _Metric):
        metrics = [DEFAULT_METRICS[metric] for metric in metrics]

    if ignore_original_features:
        dataset = dataset.copy()
        dataset.graphs = _clean_graphs(True, True, dataset.graphs)

    if features_to_test is None:
        features_to_test = _determine_features_to_test(
            node_features, edge_features, fast
        )
    else:
        features_to_test = list(features_to_test)
        for feature in features_to_test:
            if (
                feature not in VERTEX_LABELING_METHODS
                and feature not in EDGE_LABELING_METHODS
            ):
                raise ValueError(f"Unknown feature {feature}")

    has_node_features = bool(dataset.graphs[0].vertex_attributes())
    has_edge_features = bool(dataset.graphs[0].edge_attributes())

    states_to_check = ["Without node or edge features"]
    if has_node_features:
        states_to_check.append("With node features")
    if has_edge_features:
        states_to_check.append("With edge features")
    if has_node_features and has_edge_features:
        states_to_check.append("With node and edge features")

    max_feature_count = min(max_feature_count, len(features_to_test))
    results = {}
    for state in states_to_check:
        current_features_to_test = features_to_test.copy()
        previous_best_features, previous_best_result = [], None
        for feature_count in range(
            max_feature_count + 1  # we also consider the zero-feature case
        ):
            best_feature, best_result = _evaluate_features(
                dataset,
                metrics,
                current_features_to_test,
                results,
                state,
                previous_best_features,
                feature_count,
                iterations,
            )
            if best_feature is not None:
                previous_best_features.append(best_feature)
                current_features_to_test.remove(best_feature)
            if (
                all(
                    best_result.results[m.name][iterations] == m.best_value
                    for m in metrics
                )
            ) or not _result_is_better(
                best_result, previous_best_result, metrics, iterations
            ):
                # already at full efficiency or no improvement
                break
            previous_best_result = best_result

    return RecommendationResult(dataset, results, metrics, iterations)


def _evaluate_features(
    dataset,
    metrics,
    features_to_test,
    results,
    state,
    previous_best_features,
    feature_count,
    iterations,
):
    best_feature, best_result = None, None
    if feature_count == 0:
        result = evaluate(
            dataset.copy(),
            metrics=metrics,
            additional_features=None,
            iterations=iterations,
            ignore_node_features=state == "Without node or edge features"
            or state == "With edge features",
            ignore_edge_features=state == "Without node or edge features"
            or state == "With node features",
        )
        results[(state, tuple())] = result
        best_result = result
    else:
        for feature in features_to_test:
            test_dataset = dataset.copy()
            features = (feature, *previous_best_features)
            result = evaluate(
                test_dataset,
                metrics=metrics,
                additional_features=features,
                iterations=iterations,
                ignore_node_features=state == "Without node or edge features"
                or state == "With edge features",
                ignore_edge_features=state == "Without node or edge features"
                or state == "With node features",
            )
            results[(state, features)] = result
            if _result_is_better(result, best_result, metrics, iterations):
                best_feature, best_result = feature, result
    return best_feature, best_result


def _determine_features_to_test(
    node_features: bool, edge_features: bool, fast: bool
) -> List[str]:
    """Determine the methods to test for recommendation.

    Parameters
    ----------
    node_features : bool
        Whether to recommend node features.
    edge_features : bool
        Whether to recommend edge features.
    fast : bool
        Whether to only use the features that are scalable to large datasets.

    Returns
    -------
    List[str]
        The methods to test.
    """
    features_to_test = []
    if node_features:
        features_to_test += [
            method
            for method in VERTEX_LABELING_METHODS
            if not method.startswith("Marked WL hash")
        ]

    if edge_features:
        features_to_test += [
            method
            for method in EDGE_LABELING_METHODS
            if not method.startswith("Marked WL hash")
        ]
        # No need to check node-reduced edge features
        features_to_test = [
            method
            for method in features_to_test
            if not method.endswith("as node label")
        ]

    if fast:
        features_to_test = [
            method for method in features_to_test if "count of" not in method
        ]

    return features_to_test


def _result_is_better(
    result: EvaluationResult,
    best_result: EvaluationResult,
    metrics: List[str],
    iterations: int,
) -> bool:
    """Check whether a result is better than another.

    Parameters
    ----------
    result : EvaluationResult
        The result to check.
    best_result : EvaluationResult
        The best result so far.
    metrics : List[str]
        The metrics to evaluate the dataset on.
    iterations : int
        The number of iterations to run the comparison for.

    Returns
    -------
    bool
        True if the result is better than the best result, False otherwise.
    """
    if best_result is None:
        return True

    for metric in metrics:
        if metric.higher_is_better:
            if (
                result.results[metric.name][iterations]
                < best_result.results[metric.name][iterations]
            ):
                return False
        else:
            if (
                result.results[metric.name][iterations]
                > best_result.results[metric.name][iterations]
            ):
                return False
    else:
        return True

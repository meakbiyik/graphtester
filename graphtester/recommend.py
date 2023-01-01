"""Recommend additional features to add to a dataset."""
import functools

import pandas as pd

from graphtester.evaluate.dataset import EvaluationResult, evaluate
from graphtester.io.dataset import Dataset
from graphtester.label import EDGE_LABELING_METHODS, VERTEX_LABELING_METHODS


class RecommendationResult:
    """The result of a recommendation."""

    def __init__(
        self,
        dataset: Dataset,
        results: dict[tuple[str, tuple[str]], EvaluationResult],
    ):
        """Initialize a recommendation result.

        Parameters
        ----------
        dataset : Dataset
            The dataset that was recommended features for.
        results : dict[tuple[str, tuple[str]], EvaluationResult]
            The evaluation results for each recommended feature. The index
            of this dictionary is a tuple of the state of the dataset (with or
            without node or edge features) and the features that were added to
            achieve the values at the EvaluationResult.
        """
        self.dataset = dataset
        self.results = results

    def __repr__(self) -> str:
        """Return the representation of a recommendation result."""
        return f"RecommendationResult(dataset={self.dataset})"

    @functools.cache
    def as_dataframe(self) -> pd.DataFrame:
        """Return the recommendation result as a pandas dataframe."""
        rows_dict = {}
        # we will create different datasets for each state
        # before merging them into a single dataframe
        for (state, features), results in self.results.items():
            if state not in rows_dict:
                rows_dict[state] = []
            rows_dict[state].append(
                {
                    "Feature count": len(features),
                    "Feature name(s)": " + ".join(
                        [
                            f"{f} ({'n' if f in VERTEX_LABELING_METHODS else 'e'})"
                            for f in features
                        ]
                    ),
                    "Identifiability": results.identifiability[1],
                    "Upper bound accuracy": results.upper_bound_accuracy[1],
                    "Isomorphism": results.isomorphism,
                }
            )

        dataframes = []
        for idx, (state, rows) in enumerate(rows_dict.items()):
            df = pd.DataFrame(rows)
            # Drop features that are not in top 5 per feature count and state
            df = df.sort_values(
                by=["Feature count", "Identifiability"], ascending=False
            )
            df = df.groupby("Feature count").head(5).reset_index(drop=True)
            if idx != 0:
                df = df.drop(columns=["Feature count"])
            dataframes.append(df)

        states = list(rows_dict.keys())
        report = pd.concat(dataframes, axis=1, keys=states)
        report = report.set_index((states[0], "Feature count"))
        report.index.name = "Feature count"
        report.name = self.dataset.name
        report = report.round(4)
        return report

    def __str__(self) -> str:
        """Return the string representation of a recommendation result."""
        return self.as_dataframe().to_string()


def recommend(
    dataset: Dataset, max_feature_count=3, node_features=True, edge_features=True
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
    max_feature_count : int, optional
        The maximum number of features to combine into a set, by default 3
    node_features : bool, optional
        Whether to recommend node features, by default True
    edge_features : bool, optional
        Whether to recommend edge features, by default True

    Returns
    -------
    RecommendationResult
        The recommendation result.
    """
    methods_to_test = _determine_methods_to_test(node_features, edge_features)

    has_node_features = bool(dataset.graphs[0].vertex_attributes())
    has_edge_features = bool(dataset.graphs[0].edge_attributes())

    states_to_check = ["Without node or edge features"]
    if has_node_features:
        states_to_check.append("With node features")
    if has_edge_features:
        states_to_check.append("With edge features")
    if has_node_features and has_edge_features:
        states_to_check.append("With node and edge features")

    max_feature_count = min(max_feature_count, len(methods_to_test))
    results = {}
    for state in states_to_check:
        previous_best_features = []
        for _ in range(max_feature_count):
            best_feature, best_result = None, 0
            for feature in methods_to_test:
                test_dataset = dataset.copy()
                features = (feature, *previous_best_features)
                result = evaluate(
                    test_dataset,
                    additional_features=features,
                    iterations=1,
                    ignore_node_features=state == "Without node or edge features"
                    or state == "With edge features",
                    ignore_edge_features=state == "Without node or edge features"
                    or state == "With node features",
                )
                results[(state, features)] = result
                if result.identifiability[1] > best_result:
                    best_feature, best_result = feature, result.identifiability[1]
            previous_best_features.append(best_feature)
            methods_to_test.remove(best_feature)
            if best_result == 1:
                # already at full efficiency
                break

    return RecommendationResult(dataset, results)


def _determine_methods_to_test(node_features: bool, edge_features: bool) -> list[str]:
    """Determine the methods to test for recommendation.

    Parameters
    ----------
    node_features : bool
        Whether to recommend node features.
    edge_features : bool
        Whether to recommend edge features.

    Returns
    -------
    list[str]
        The methods to test.
    """
    methods_to_test = []
    if node_features:
        methods_to_test += [
            method
            for method in VERTEX_LABELING_METHODS
            if not method.startswith("Marked WL hash")
        ]

    if edge_features:
        methods_to_test += [
            method
            for method in EDGE_LABELING_METHODS
            if not method.startswith("Marked WL hash")
        ]
        # No need to check node-reduced edge features
        methods_to_test = [
            method for method in methods_to_test if not method.endswith("as node label")
        ]

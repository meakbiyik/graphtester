"""Demo script for the graphtester package."""
import multiprocessing as mp
import pickle

import graphtester as gt
from graphtester.evaluate.dataset import DEFAULT_METRICS, _Metric
from graphtester.io.dataset import Dataset
from graphtester.io.load import DATASETS

MULTIPROCESSING = True

datasets_to_evaluate = [
    dataset for dataset in DATASETS if not dataset.startswith("GT")
][-6:-5]


def select_metric(dataset: Dataset) -> _Metric:
    """Select the metric to use for a dataset."""
    suffix = None
    labels = None
    if dataset.edge_labels is not None:
        suffix = "_edge"
        labels = [label for edge_labels in dataset.edge_labels for label in edge_labels]
    elif dataset.node_labels is not None:
        suffix = "_node"
        labels = [label for node_labels in dataset.node_labels for label in node_labels]
    elif dataset.labels is not None:
        suffix = ""
        labels = dataset.labels

    if labels is not None:
        first_twenty_labels = labels[:100]
        labels_are_round = [round(label) == label for label in first_twenty_labels]
        if all(labels_are_round):
            return DEFAULT_METRICS[f"upper_bound_accuracy{suffix}"]
        else:
            return DEFAULT_METRICS[f"lower_bound_mse{suffix}"]


def analyze_dataset(dataset_name: str):
    """Analyze a dataset."""
    print(f"Dataset: {dataset_name}")

    # Load the dataset
    dataset = gt.load(dataset_name)
    print(dataset)

    metrics = [select_metric(dataset)]
    print(metrics[0].description)

    has_node_features = len(dataset.graphs[0].vs.attributes()) > 0
    has_edge_features = len(dataset.graphs[0].es.attributes()) > 0

    states = ["without_features"]
    if has_node_features:
        states.append("with_node_features")
    if has_edge_features:
        states.append("with_edge_features")
    if has_node_features and has_edge_features:
        states.append("with_node_and_edge_features")

    is_regression = metrics[0].name.startswith("lower_bound")

    iterations = 10

    for state in states:
        # Evaluate the dataset
        evaluation = gt.evaluate(
            dataset,
            metrics=metrics,
            ignore_node_features=state in ["without_features", "with_edge_features"],
            ignore_edge_features=state in ["without_features", "with_node_features"],
            iterations=iterations,
        )
        # pickle the evaluation
        with open(f"evaluation_{dataset_name}_{state}_{is_regression}.pickle", "wb") as f:
            pickle.dump(evaluation, f)

        print(evaluation)

    if evaluation.results[metrics[0].name][0] == metrics[0].best_value:
        print(f"Dataset is fully solvable in 0 iterations.")
    else:
        # Recommend features to add to the dataset
        recommendation = gt.recommend(
            dataset,
            metrics=metrics,
            max_feature_count=1,
            node_features=True,
            edge_features=True,
            iterations=iterations,
            fast=True,
        )

        # pickle the recommendation
        with open(f"recommendation_{dataset_name}_{is_regression}.pickle", "wb") as f:
            pickle.dump(recommendation, f)

        print(recommendation)


def analyze_dataset_with_skip(dataset_name: str):
    """Analyze a dataset with skip."""
    try:
        analyze_dataset(dataset_name)
    except Exception as e:
        print(f"Dataset: {dataset_name}, Error: {e}")


if __name__ == "__main__":

    if MULTIPROCESSING:
        with mp.Pool() as pool:
            pool.map(analyze_dataset_with_skip, datasets_to_evaluate)

    else:
        for dataset_name in datasets_to_evaluate:
            analyze_dataset(dataset_name)

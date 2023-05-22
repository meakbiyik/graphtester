"""Demo script for the graphtester package."""
import multiprocessing as mp
import pickle

import graphtester as gt
from graphtester.evaluate.dataset import DEFAULT_METRICS, _Metric
from graphtester.io.dataset import Dataset
from graphtester.io.load import DATASETS

MULTIPROCESSING = True
ONLY_RECOMMENDATION = True
WITH_ORIGINAL_FEATS = True
FEATURES_TO_TEST = [
    "Eigenvector centrality",
    "Eccentricity",
    "Local transitivity",
    "Harmonic centrality",
    "Closeness centrality",
    "Burt's constraint",
    "Betweenness centrality",
]
GRAPH_COUNT = 10000 # If the dataset has more graphs than this, it is subsampled
ITERATIONS = 1

datasets_to_skip = ["GT", "GT-small", "ZINC_FULL"]
datasets_to_evaluate = [dataset for dataset in DATASETS if dataset not in datasets_to_skip]

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
        first_hundred_labels = labels[:100]
        labels_are_round = [round(label) == label for label in first_hundred_labels]
        if all(labels_are_round):
            return DEFAULT_METRICS[f"upper_bound_accuracy{suffix}"]
        else:
            return DEFAULT_METRICS[f"lower_bound_mse{suffix}"]


def analyze_dataset(dataset_name: str):
    """Analyze a dataset."""
    print(f"Dataset: {dataset_name}", flush=True)

    # Load the dataset
    dataset = gt.load(dataset_name, graph_count=GRAPH_COUNT)
    print(dataset, flush=True)

    metrics = [select_metric(dataset)]
    print(metrics[0].description, flush=True)

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

    if not ONLY_RECOMMENDATION:
        for state in states:
            # Evaluate the dataset
            evaluation = gt.evaluate(
                dataset,
                metrics=metrics,
                ignore_node_features=state
                in ["without_features", "with_edge_features"],
                ignore_edge_features=state
                in ["without_features", "with_node_features"],
                iterations=ITERATIONS,
            )
            # pickle the evaluation
            with open(
                f"evaluation_{dataset_name}_{state}_{is_regression}.pickle", "wb"
            ) as f:
                pickle.dump(evaluation, f)

            print(evaluation, flush=True)

    recommender_filename = (
        f"recommendation_{dataset_name}{'_regression' if is_regression else '_classification'}"
        f"{'_without_original_feats' if not WITH_ORIGINAL_FEATS else ''}"
        f"_{ITERATIONS}_iter"
        f"{f'_{GRAPH_COUNT}' if GRAPH_COUNT is not None else ''}.pickle"
    )

    # Check if the recommendation already exists, if so, skip
    try:
        with open(recommender_filename, "rb") as f:
            recommendation = pickle.load(f)
        print(f"Recommendation already exists for {dataset_name}", flush=True)
        print(recommendation, flush=True)
        return
    except FileNotFoundError:
        pass

    # Recommend features to add to the dataset
    recommendation = gt.recommend(
        dataset,
        metrics=metrics,
        max_feature_count=1,
        features_to_test=FEATURES_TO_TEST,
        ignore_original_features=not WITH_ORIGINAL_FEATS,
        iterations=ITERATIONS,
    )

    # pickle the recommendation
    with open(recommender_filename,
        "wb",
    ) as f:
        pickle.dump(recommendation, f)

    print(recommendation, flush=True)


def analyze_dataset_with_skip(dataset_name: str):
    """Analyze a dataset with skip."""
    try:
        analyze_dataset(dataset_name)
    except Exception as e:
        print(f"Dataset: {dataset_name}, Error: {e}", flush=True)


if __name__ == "__main__":

    if MULTIPROCESSING:
        with mp.Pool() as pool:
            pool.map(analyze_dataset_with_skip, datasets_to_evaluate)

    else:
        for dataset_name in datasets_to_evaluate:
            analyze_dataset(dataset_name)

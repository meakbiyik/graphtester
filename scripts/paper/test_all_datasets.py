"""Demo script for the graphtester package."""
import multiprocessing as mp
import pickle
import os

import graphtester as gt
from graphtester.evaluate.dataset import DEFAULT_METRICS, _Metric
from graphtester.io.dataset import Dataset
from graphtester.io.load import DATASETS

MULTIPROCESSING = True
PROCESS_COUNT = 4
ONLY_RECOMMENDATION = False
ONLY_EVALUATION = True
WITH_ORIGINAL_FEATS = True
FEATURES_TO_TEST = [
    "Laplacian positional encoding (dim=1)",
    "Laplacian positional encoding (dim=2)",
    "Laplacian positional encoding (dim=4)",
    "Laplacian positional encoding (dim=8)",
    "Laplacian positional encoding (dim=16)",
    "Laplacian positional encoding (dim=32)",
]

GRAPH_COUNT = 5000 # If the dataset has more graphs than this, it is subsampled
ITERATIONS = 3
USE_NODE_HASHES_FOR_GRAPH = False
ONLY_EVALUATE_NON_GRAPH_TASKS = True
USE_TRANSFORMER_BASIS = False

# print all the settings
print("Settings:", flush=True)
print(f"Multiprocessing: {MULTIPROCESSING}", flush=True)
print(f"Only recommendation: {ONLY_RECOMMENDATION}", flush=True)
print(f"Only evaluation: {ONLY_EVALUATION}", flush=True)
print(f"With original features: {WITH_ORIGINAL_FEATS}", flush=True)
print(f"Graph count: {GRAPH_COUNT}", flush=True)
print(f"Iterations: {ITERATIONS}", flush=True)
print(f"Use node hashes for graph: {USE_NODE_HASHES_FOR_GRAPH}", flush=True)
print(f"Only evaluate non-graph tasks: {ONLY_EVALUATE_NON_GRAPH_TASKS}", flush=True)
print(f"Use transformer basis: {USE_TRANSFORMER_BASIS}", flush=True)

if MULTIPROCESSING:
    print("Using multiprocessing", flush=True)
    print(f"Number of processes: {PROCESS_COUNT}", flush=True)

datasets_to_skip = ["GT", "GT-small", "ZINC_FULL"]
if USE_NODE_HASHES_FOR_GRAPH:
    datasets_to_evaluate = ["ZINC", "MNIST", "CIFAR10"]
else:
    datasets_to_evaluate = [
        dataset for dataset in DATASETS if dataset not in datasets_to_skip
    ]

datasets_to_evaluate = [
    "PascalVOC-SP",
    "COCO-SP",
    "PCQM-Contact",
    "Peptides-func",
    "Peptides-struct",
]

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
        suffix = "_graph_node" if USE_NODE_HASHES_FOR_GRAPH else ""
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

    # Load the dataset
    dataset = gt.load(dataset_name, graph_count=GRAPH_COUNT)

    metrics = [select_metric(dataset)]

    if ONLY_EVALUATE_NON_GRAPH_TASKS and not ("node" in metrics[0].type or "edge" in metrics[0].type):
        return

    print(f"Dataset: {dataset_name}", flush=True)
    print(dataset, flush=True)
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

    if not ONLY_RECOMMENDATION or ONLY_EVALUATION:
        for state in states:
            evaluation_filename = (
                f"evaluation_{dataset_name}_{state}_{is_regression}"
                f"{'_transformer' if USE_TRANSFORMER_BASIS else ''}"
                "_final_fixed.pickle"
            )

            # Check if the evaluation already exists, if so, skip
            try:
                with open(evaluation_filename, "rb") as f:
                    evaluation = pickle.load(f)
                print(f"Evaluation already exists for {dataset_name} {state}", flush=True)
                print(evaluation, flush=True)
                continue
            except FileNotFoundError:
                pass

            # Evaluate the dataset
            evaluation = gt.evaluate(
                dataset,
                metrics=metrics,
                ignore_node_features=state
                in ["without_features", "with_edge_features"],
                ignore_edge_features=state
                in ["without_features", "with_node_features"],
                iterations=ITERATIONS,
                transformer=USE_TRANSFORMER_BASIS,
            )
            # pickle the evaluation
            with open(evaluation_filename, "wb") as f:
                pickle.dump(evaluation, f)

            print(f"Evaluation for {dataset_name} {state}", flush=True)
            print(evaluation, flush=True)

    if not ONLY_EVALUATION or ONLY_RECOMMENDATION:
        recommender_filename = (
            f"recommendation_{dataset_name}{'_regression' if is_regression else '_classification'}"
            f"{'_without_original_feats' if not WITH_ORIGINAL_FEATS else ''}"
            f"_{ITERATIONS}_iter"
            f"{'_with_node_hashes' if USE_NODE_HASHES_FOR_GRAPH else ''}"
            f"{'_transformer' if USE_TRANSFORMER_BASIS else ''}"
            f"{f'_{GRAPH_COUNT}' if GRAPH_COUNT is not None else ''}_final_zinc.pickle"
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

        print(f"Recommendation for {dataset_name}", flush=True)
        print(f"Recommendation saved to {recommender_filename}", flush=True)
        print(recommendation, flush=True)


def analyze_dataset_with_skip(dataset_name: str):
    """Analyze a dataset with skip."""
    try:
        analyze_dataset(dataset_name)
    except Exception as e:
        print(f"Dataset: {dataset_name}, Error: {e}", flush=True)


if __name__ == "__main__":

    if MULTIPROCESSING:
        with mp.Pool(PROCESS_COUNT) as pool:
            pool.map(analyze_dataset_with_skip, datasets_to_evaluate)

    else:
        for dataset_name in datasets_to_evaluate:
            analyze_dataset(dataset_name)

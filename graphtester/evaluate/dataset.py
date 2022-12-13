"""Evaluate a Dataset."""

from typing import Dict, List

import igraph as ig
import pandas as pd

from graphtester.io.dataset import Dataset
from graphtester.label import label
from graphtester.test import weisfeiler_lehman_hash


class EvaluationResult:
    """Evaluation result object."""

    def __init__(
        self,
        dataset: Dataset,
        identifiability: Dict[int, float] = None,
        upper_bound_accuracy: Dict[int, float] = None,
        isomorphism: float = None,
    ):
        """Initialize an EvaluationResult object.

        Parameters
        ----------
        dataset : Dataset
            The dataset that was evaluated.
        """
        self.dataset = dataset
        self._dataset_is_labeled = dataset.is_labeled
        self.identifiability = identifiability
        self.upper_bound_accuracy = upper_bound_accuracy
        self.isomorphism = isomorphism

    def __repr__(self):
        """Return a string representation of the object."""
        return f"EvaluationResult({self.dataset.name})"

    def __str__(self):
        """Create and return a tabular report of the evaluation.

        Neatly formats the evaluation results in a tabular format with table
        headers and rows, as well as borders. The table is returned as a string.
        """
        report = pd.DataFrame(
            {
                "Dataset": self.dataset.name,
                "Identifiability": self.identifiability,
                "Upper Bound Accuracy": self.upper_bound_accuracy,
                "Isomorphism": self.isomorphism,
            }
        )
        report = report.set_index("Dataset")
        report = report.round(2)
        return report.to_string()


def evaluate(
    dataset: Dataset,
    ignore_features: bool = False,
    additional_features: List[str] = None,
    metrics: List[str] = None,
) -> EvaluationResult:
    """Evaluate a dataset.

    This method analyzes how "hard" a dataset is to classify from 1-WL perspective.
    It does so by computing the following metrics:
        1. 1-WL-identifiability: the percentage of graphs that can be uniquely
            identified with 1-WL algorithm at k iterations.
        2. Upper bound accuracy: the upper bound accuracy of 1-WL algorithm at
            k iterations for labeled datasets, considering majority vote.
        3. Isomorphism: the percentage of graphs that are isomorphic to some
            other graph in the dataset. This is a measure of how much the
            dataset is redundant.

    The analysis admits additional structural features, which are computed for each
    graph and added as node and/or edge feature.

    Parameters
    ----------
    dataset : Dataset
        The dataset to evaluate.
    ignore_features : bool, optional
        Whether to ignore the existing node and edge features of the graphs,
        by default False
    additional_features : List[str], optional
        Additional structural features to estimate and evaluate the dataset with.
        If None (default), only the existing features of the graphs are used.
    metrics : List[str], optional
        The metrics to use for evaluation. If None (default), all metrics are used.
        Currently, the following metrics are supported:
            - "identifiability": 1-WL-identifiability
            - "upper_bound_accuracy": Upper bound accuracy
            - "isomorphism": Isomorphism

    Returns
    -------
    result : EvaluationResult
        The evaluation results object.
    """
    if metrics is None:
        metrics = ["identifiability", "upper_bound_accuracy", "isomorphism"]

    graphs = dataset.graphs
    if ignore_features:
        graphs = [
            ig.Graph(
                n=graph.vcount(),
                edges=graph.get_edgelist(),
                directed=graph.is_directed(),
            )
            for graph in graphs
        ]
    if additional_features is not None:
        graphs = [label(graph, methods=additional_features) for graph in graphs]

    labels = dataset.labels
    hashes = _estimate_hashes_at_k_iterations(dataset)

    identifiability = None
    upper_bound_accuracy = None
    isomorphism = None

    if "identifiability" in metrics:
        identifiability = _evaluate_identifiability(hashes)

    if "upper_bound_accuracy" in metrics and labels is not None:
        upper_bound_accuracy = _evaluate_upper_bound_accuracy(hashes, labels)

    if "isomorphism" in metrics:
        isomorphism = _evaluate_isomorphism(dataset)

    result = EvaluationResult(
        dataset, identifiability, upper_bound_accuracy, isomorphism
    )

    return result


def _estimate_hashes_at_k_iterations(
    dataset: Dataset,
) -> dict[int, List[str]]:
    """Estimate the 1-WL hashes of a dataset at different iterations.

    We simply run 1-WL on all graphs for k iterations and count the
    number of graphs that have a unique 1-WL representation at each iteration.

    Parameters
    ----------
    dataset : Dataset
        The dataset to estimate the hashes of.
    k : int
        The number of iterations to run 1-WL.

    Returns
    -------
    hashes : dict[int, List[str]]
        The estimated hashes of the graphs.
    """
    k = 1
    hashes = {}
    stabilized_graphs = set()
    last_graph_refinements = dataset.graphs
    while len(stabilized_graphs) < len(dataset.graphs):
        new_hashes, new_graph_refinements = [], last_graph_refinements
        for idx, graph in enumerate(last_graph_refinements):
            if idx in stabilized_graphs:
                new_hashes.append(hashes[k - 1][idx])
                continue
            # Check if 1-WL has stabilized
            if k > 2 and hashes[k - 1][idx] == hashes[k - 2][idx]:
                new_hashes.append(hashes[k - 1][idx])
                stabilized_graphs.add(idx)
                continue
            edge_attrs = graph.es.attributes()
            node_attrs = graph.vs.attributes()
            # Do a single iteration of 1-WL
            graph_hash, refined_graph = weisfeiler_lehman_hash(
                graph, edge_attrs, node_attrs, k=1, return_graph=True
            )
            new_graph_refinements[idx] = refined_graph
            new_hashes.append(graph_hash)
        hashes[k] = new_hashes
        last_graph_refinements = new_graph_refinements
        k += 1
    return hashes


def _evaluate_identifiability(
    hashes: dict[int, List[str]],
) -> dict:
    """Evaluate the 1-WL-identifiability of a dataset.

    This is the percentage of graphs that can be uniquely identified with 1-WL
    algorithm at different iterations.

    Parameters
    ----------
    hashes : dict[int, List[str]]
        The estimated hashes of the graphs at different iterations.

    Returns
    -------
    identifiability : dict[int, float]
        The identifiability of the dataset.
    """
    identifiability = {}
    for k, hashes_at_k in hashes.items():
        unique_hashes = set(hashes_at_k)
        identifiability[k] = len(unique_hashes) / len(hashes_at_k)
    return identifiability


def _evaluate_upper_bound_accuracy(
    hashes: dict[int, List[str]],
    labels: List[int],
) -> dict:
    """Evaluate the upper bound accuracy of a dataset.

    This algorithm estimates the upper bound accuracy of a dataset by first
    computing the majority vote of the labels of the graphs that have the same
    1-WL representation at different iterations. For these majority labelse,
    the accuracy is then computed.

    Parameters
    ----------
    hashes : dict[int, List[str]]
        The estimated hashes of the graphs at different iterations.
    labels : List[int]
        The labels of the graphs in the dataset.

    Returns
    -------
    upper_bound_accuracy : dict[int, float]
        The upper bound accuracy of the dataset.
    """
    upper_bound_accuracy = {}
    for k, hashes_at_k in hashes.items():
        majority_labels = {}
        for hash, lbl in zip(hashes_at_k, labels):
            if hash not in majority_labels:
                majority_labels[hash] = []
            majority_labels[hash].append(lbl)
        majority_labels = {
            hash: max(set(labels), key=labels.count)
            for hash, labels in majority_labels.items()
        }
        predicted_labels = [majority_labels[hash] for hash in hashes_at_k]
        upper_bound_accuracy[k] = sum(
            [y_true == y_pred for y_true, y_pred in zip(labels, predicted_labels)]
        ) / len(labels)
    return upper_bound_accuracy


def _evaluate_isomorphism(
    dataset: Dataset,
) -> float:
    """Evaluate the isomorphism of a dataset.

    We simply calculate the actual percentage of non-isomorphic graphs in the
    dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to evaluate.

    Returns
    -------
    isomorphism : float
        The isomorphism of the dataset.
    """
    non_isomorphic_graphs = []
    for graph in dataset.graphs:
        for other_graph in non_isomorphic_graphs:
            if graph.isomorphic(other_graph):
                break
        else:
            non_isomorphic_graphs.append(graph)
    return len(non_isomorphic_graphs) / len(dataset.graphs)

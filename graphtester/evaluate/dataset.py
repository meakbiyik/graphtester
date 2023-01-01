"""Evaluate a Dataset."""
import functools
from collections import Counter
from typing import Dict, List, Optional

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

    @functools.cache
    def as_dataframe(self) -> pd.DataFrame:
        """Create and return a tabular report of the evaluation."""
        report = pd.DataFrame(
            {
                "Identifiability": self.identifiability,
                "Upper Bound Accuracy": self.upper_bound_accuracy,
                "Isomorphism": self.isomorphism,
            }
        )
        report.index.rename("Iteration", inplace=True)
        report.name = self.dataset.name
        report = report.round(4) * 100
        return report

    def __str__(self):
        """Create and return a tabular report of the evaluation."""
        return self.as_dataframe().to_string()


def evaluate(
    dataset: Dataset,
    ignore_node_features: bool = False,
    ignore_edge_features: bool = True,
    additional_features: List[str] = None,
    metrics: List[str] = None,
    iterations: int = 3,
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
    ignore_node_features : bool, optional
        Whether to ignore the existing node and edge features of the graphs,
        by default False
    ignore_edge_features : bool, optional
        Whether to ignore the existing edge features of the graphs, by default True
    additional_features : List[str], optional
        Additional structural features to estimate and evaluate the dataset with.
        If None (default), only the existing features of the graphs are used.
    metrics : List[str], optional
        The metrics to use for evaluation. If None (default), all metrics are used.
        Currently, the following metrics are supported:
            - "identifiability": 1-WL-identifiability
            - "upper_bound_accuracy": Upper bound accuracy
            - "isomorphism": Isomorphism
    iterations : int, optional
        The number of iterations to run 1-WL for, by default 3

    Returns
    -------
    result : EvaluationResult
        The evaluation results object.
    """
    if metrics is None:
        metrics = ["identifiability", "upper_bound_accuracy", "isomorphism"]

    graphs = dataset.graphs
    if ignore_node_features or ignore_edge_features:
        graphs = [
            ig.Graph(
                n=graph.vcount(),
                edges=graph.get_edgelist(),
                directed=graph.is_directed(),
                vertex_attrs={
                    attr: graph.vs[attr] for attr in graph.vertex_attributes()
                }
                if not ignore_node_features
                else {},
                edge_attrs={attr: graph.es[attr] for attr in graph.edge_attributes()}
                if not ignore_edge_features
                else {},
            )
            for graph in graphs
        ]
    if additional_features is not None:
        graphs = [label(graph, methods=additional_features) for graph in graphs]

    labels = dataset.labels
    hashes = _estimate_hashes_at_k_iterations(graphs.copy(), iterations)

    identifiability = None
    upper_bound_accuracy = None
    isomorphism = None

    if "identifiability" or "isomorphism" in metrics:
        isomorphism_list = _get_isomorphism_list(graphs)

    if "identifiability" in metrics:
        identifiability = _evaluate_identifiability(hashes, isomorphism_list)

    if "upper_bound_accuracy" in metrics and labels is not None:
        upper_bound_accuracy = _evaluate_upper_bound_accuracy(hashes, labels)

    if "isomorphism" in metrics:
        isomorphism = _evaluate_isomorphism(isomorphism_list)

    result = EvaluationResult(
        dataset, identifiability, upper_bound_accuracy, isomorphism
    )

    return result


def _estimate_hashes_at_k_iterations(
    graphs: List[ig.Graph],
    iterations: int = 3,
) -> dict[int, List[str]]:
    """Estimate the 1-WL hashes of a dataset at different iterations.

    We simply run 1-WL on all graphs for k iterations and count the
    number of graphs that have a unique 1-WL representation at each iteration.

    Parameters
    ----------
    graphs : List[ig.Graph]
        The graphs to estimate the hashes of.
    iterations : int, optional
        The number of iterations to run 1-WL for, by default 3

    Returns
    -------
    hashes : dict[int, List[str]]
        The estimated hashes of the graphs.
    """
    k = 1
    hashes = {}
    stabilized_graphs = set()
    graph_count = len(graphs)
    last_graph_refinements = graphs
    last_graph_hashes = [None] * graph_count
    while len(stabilized_graphs) < graph_count and k <= iterations:
        new_hashes, new_graph_refinements = last_graph_hashes, last_graph_refinements
        for idx, graph in enumerate(last_graph_refinements):
            if idx in stabilized_graphs:
                continue
            edge_attrs = graph.es.attributes()
            node_attrs = graph.vs.attributes() if k == 1 else "label"
            # Do a single iteration of 1-WL
            graph_hash, refined_graph = weisfeiler_lehman_hash(
                graph, edge_attrs, node_attrs, iterations=1, return_graph=True
            )
            # Check if 1-WL has stabilized
            if k > 2 and graph_hash == last_graph_hashes[idx]:
                stabilized_graphs.add(idx)
            new_graph_refinements[idx] = refined_graph
            new_hashes[idx] = graph_hash
        hashes[k] = new_hashes.copy()
        last_graph_refinements = new_graph_refinements
        last_graph_hashes = new_hashes
        k += 1
    return hashes


def _get_isomorphism_list(graphs: List[ig.Graph]) -> List[Optional[int]]:
    """Compute the isomorphism list of a dataset.

    An "isomorphism list" as referred in this package is a list where each
    element is the index of the first graph that is isomorphic to in the
    dataset. If no graph was isomorphic to a graph in the list, the element is None.

    Parameters
    ----------
    graphs : List[ig.Graph]
        The graphs to compute the isomorphism list for.

    Returns
    -------
    isomorphism_list : List[Optional[int]]
        The isomorphism list of the dataset.
    """
    node_colors = [None] * len(graphs)
    node_attributes = graphs[0].vs.attributes()
    if len(node_attributes) > 0:
        nodeattr_hashset = set()
        for i in range(len(graphs)):
            # estimate the node colors for each node
            # by looping over all node features
            # since hashes are too big to fit to C long, we need to map them later
            node_attrs = [graphs[i].vs[attr] for attr in node_attributes]
            node_hashes = [
                tuple(str(a) for a in attrtuple) for attrtuple in zip(*node_attrs)
            ]
            nodeattr_hashset.update(node_hashes)
            node_colors[i] = node_hashes
        nodeattr_hashmap = {h: i for i, h in enumerate(nodeattr_hashset)}
        for i in range(len(graphs)):
            node_colors[i] = [nodeattr_hashmap[h] for h in node_colors[i]]

    edge_colors = [None] * len(graphs)
    edge_attributes = graphs[0].es.attributes()
    if len(edge_attributes) > 0:
        edge_hashset = set()
        for i in range(len(graphs)):
            # estimate the node colors for each node
            # by looping over all node features
            # since hashes are too big to fit to C long, we need to map them later
            edge_attrs = [graphs[i].es[attr] for attr in edge_attributes]
            edge_hashes = [
                tuple(str(a) for a in attrtuple) for attrtuple in zip(*edge_attrs)
            ]
            edge_hashset.update(edge_hashes)
            edge_colors[i] = edge_hashes
        edge_hashmap = {h: i for i, h in enumerate(edge_hashset)}
        for i in range(len(graphs)):
            edge_colors[i] = [edge_hashmap[h] for h in edge_colors[i]]

    # If one graph is isomorphic to another, we only need to compute the
    # isomorphism for one of them, for a new graph. So, we keep track of the
    # first graph that is isomorphic to each graph, and skip the rest.
    isomorphism_list = [None] * len(graphs)
    for i in range(len(graphs)):
        for j in range(i):
            if isomorphism_list[i] is None and graphs[j].isomorphic_vf2(
                graphs[i],
                color1=node_colors[j],
                color2=node_colors[i],
                edge_color1=edge_colors[j],
                edge_color2=edge_colors[i],
            ):
                isomorphism_list[i] = j
                break

    return isomorphism_list


def _evaluate_identifiability(
    hashes: dict[int, List[str]],
    isomorphism_list: List[Optional[int]],
) -> dict:
    """Evaluate the 1-WL-identifiability of a dataset.

    This is the percentage of non-isomorphic graphs that can be uniquely identified
    with 1-WL algorithm at different iterations.

    Why this version, instead of just estimating the percentage of unique hashes?
    We simply follow the implementation of the "1-WL Expressiveness Is (Almost)
    All You Need" paper: "The fraction of non-isomorphic graphs [...] that can be
    identified by 1-WL representations."

    Parameters
    ----------
    hashes : dict[int, List[str]]
        The estimated hashes of the graphs at different iterations.
    isomorphism_list : List[Optional[int]]
        The isomorphism list of the dataset.

    Returns
    -------
    identifiability : dict[int, float]
        The identifiability of the dataset.
    """
    isomorphic_graph_mask = [idx is None for idx in isomorphism_list]
    for idx in isomorphism_list:
        if idx is not None:
            isomorphic_graph_mask[idx] = False

    identifiability = {}
    for k, hashes_at_k in hashes.items():
        noniso_graph_hashes = [
            hsh for hsh, iso in zip(hashes_at_k, isomorphic_graph_mask) if iso
        ]
        hash_counts = Counter(noniso_graph_hashes)
        unique_hashes = [h for h, count in hash_counts.items() if count == 1]
        identifiability[k] = len(unique_hashes) / len(noniso_graph_hashes)
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
    isomorphism_list: List[Optional[int]],
) -> float:
    """Evaluate the isomorphism of a dataset through the isomorphism list.

    We simply calculate the percentage of graphs that are not isomorphic to
    any other graph in the dataset.

    As defined in the previous paper: "how large the fraction of unique
    graphs is, i.e. for how many graphs no other isomorphic graph is
    present in the dataset"

    Parameters
    ----------
    isomorphism_list : List[Optional[int]]
        The isomorphism list of the dataset.

    Returns
    -------
    isomorphism : float
        The isomorphism of the dataset.
    """
    # We can't just count the number of None's in the isomorphism list, we
    # also need to remove any graphs that are referenced by any number in
    # the list.
    isomorphic_graph_mask = [idx is None for idx in isomorphism_list]
    for idx in isomorphism_list:
        if idx is not None:
            isomorphic_graph_mask[idx] = False

    return sum(isomorphic_graph_mask) / len(isomorphism_list)

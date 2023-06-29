"""Evaluate a Dataset."""
from collections import Counter
from typing import Dict, List, Literal, Optional, Tuple, Union

import igraph as ig
import pandas as pd

from graphtester.io.dataset import Dataset
from graphtester.label import label
from graphtester.test import weisfeiler_lehman_hash


class _Metric:
    """A metric to evaluate a dataset."""

    def __init__(
        self,
        name: str,
        description: str,
        higher_is_better: bool,
        type: Literal["graph", "graph_node", "node", "edge"],
        evaluator: callable,
        best_value: Optional[float] = None,
    ):
        """Initialize a Metric object.

        Parameters
        ----------
        name : str
            The name of the metric.
        """
        self.name = name
        self.description = description
        self.higher_is_better = higher_is_better
        self.type = type
        self._evaluator = evaluator
        self.best_value = best_value

    def evaluate(
        self,
        graphs: List[ig.Graph],
        labels: List[float],
        node_labels: List[List[float]],
        edge_labels: List[List[float]],
        hashes: Dict[int, List[str]],
        node_hashes: Dict[int, List[List[str]]],
        edge_hashes: Dict[int, List[List[str]]],
    ):
        if self.type == "graph":
            return self._evaluator(graphs, hashes, labels)
        elif self.type == "graph_node":
            # Flatten the node hashes but use graph labels with it
            # by repeating the graph labels for each node
            # this is still graph-level evaluation
            flat_node_hashes = {
                graph_index: [hsh for hshs in hashes_list for hsh in hshs]
                for graph_index, hashes_list in node_hashes.items()
            }
            repeated_graph_labels = [
                graph_label
                for graph, graph_label in zip(graphs, labels)
                for _ in range(graph.vcount())
            ]
            return self._evaluator(graphs, flat_node_hashes, repeated_graph_labels)
        elif self.type == "node":
            flat_node_hashes = {
                graph_index: [hsh for hshs in hashes_list for hsh in hshs]
                for graph_index, hashes_list in node_hashes.items()
            }
            flat_node_labels = [lbl for lbls in node_labels for lbl in lbls]
            return self._evaluator(graphs, flat_node_hashes, flat_node_labels)
        elif self.type == "edge":
            flat_edge_hashes = {
                graph_index: [hsh for hshs in hashes_list for hsh in hshs]
                for graph_index, hashes_list in edge_hashes.items()
            }
            flat_edge_labels = [lbl for lbls in edge_labels for lbl in lbls]
            return self._evaluator(graphs, flat_edge_hashes, flat_edge_labels)


def _estimate_identifiability(graphs, hashes, labels):
    isomorphism_list = _get_isomorphism_list(graphs)
    identifiability = _evaluate_identifiability(hashes, isomorphism_list)
    return identifiability


def _estimate_upper_bound_accuracy(graphs, hashes, labels):
    int_labels = [int(label) for label in labels]
    upper_bound_accuracy = _evaluate_upper_bound_accuracy(hashes, int_labels)
    return upper_bound_accuracy


def _estimate_mse(graphs, hashes, labels):
    mse = _evaluate_mse(hashes, labels)
    return mse


def _estimate_f1(graphs, hashes, labels):
    int_labels = [int(label) for label in labels]
    f1 = _evaluate_f1(hashes, int_labels)
    return f1


DEFAULT_METRICS = {
    "identifiability": _Metric(
        "identifiability",
        "1-WL-Identifiability",
        True,
        "graph",
        _estimate_identifiability,
        1.0,
    ),
    "upper_bound_accuracy": _Metric(
        "upper_bound_accuracy",
        "Upper Bound Accuracy",
        True,
        "graph",
        _estimate_upper_bound_accuracy,
        1.0,
    ),
    "upper_bound_accuracy_graph_node": _Metric(
        "upper_bound_accuracy_graph_node",
        "Upper Bound Accuracy (Graph - Node)",
        True,
        "graph_node",
        _estimate_upper_bound_accuracy,
        1.0,
    ),
    "upper_bound_accuracy_node": _Metric(
        "upper_bound_accuracy_node",
        "Upper Bound Accuracy (Node)",
        True,
        "node",
        _estimate_upper_bound_accuracy,
        1.0,
    ),
    "upper_bound_accuracy_edge": _Metric(
        "upper_bound_accuracy_edge",
        "Upper Bound Accuracy (Edge)",
        True,
        "edge",
        _estimate_upper_bound_accuracy,
        1.0,
    ),
    "lower_bound_mse": _Metric(
        "lower_bound_mse", "Lower Bound MSE", False, "graph", _estimate_mse, 0.0
    ),
    "lower_bound_mse_graph_node": _Metric(
        "lower_bound_mse_graph_node",
        "Lower Bound MSE (Graph - Node)",
        False,
        "graph_node",
        _estimate_mse,
        0.0,
    ),
    "lower_bound_mse_node": _Metric(
        "lower_bound_mse_node",
        "Lower Bound MSE (Node)",
        False,
        "node",
        _estimate_mse,
        0.0,
    ),
    "lower_bound_mse_edge": _Metric(
        "lower_bound_mse_edge",
        "Lower Bound MSE (Edge)",
        False,
        "edge",
        _estimate_mse,
        0.0,
    ),
    "upper_bound_f1_micro": _Metric(
        "upper_bound_f1_micro",
        "Upper Bound F1-micro",
        True,
        "graph",
        _estimate_f1,
        1.0,
    ),
    "upper_bound_f1_micro_graph_node": _Metric(
        "upper_bound_f1_micro_graph_node",
        "Upper Bound F1-micro (Graph - Node)",
        True,
        "graph_node",
        _estimate_f1,
        1.0,
    ),
    "upper_bound_f1_micro_node": _Metric(
        "upper_bound_f1_micro_node",
        "Upper Bound F1-micro (Node)",
        True,
        "node",
        _estimate_f1,
        1.0,
    ),
    "upper_bound_f1_micro_edge": _Metric(
        "upper_bound_f1_micro_edge",
        "Upper Bound F1-micro (Edge)",
        True,
        "edge",
        _estimate_f1,
        1.0,
    ),
}


class EvaluationResult:
    """Evaluation result object."""

    def __init__(
        self,
        dataset: Dataset,
        metrics: List[_Metric],
        results: Dict[str, Dict[int, float]],
    ):
        """Initialize an EvaluationResult object.

        Parameters
        ----------
        dataset : Dataset
            The dataset that was evaluated.
        """
        self.dataset_name = dataset.name
        self.metrics = metrics
        self.results = results
        self._dataframe = None

    def __repr__(self):
        """Return a string representation of the object."""
        return f"EvaluationResult({self.dataset_name})"

    def as_dataframe(self) -> pd.DataFrame:
        """Create and return a tabular report of the evaluation."""
        if self._dataframe is not None:
            return self._dataframe
        data = {}
        for metric in self.metrics:
            data[metric.description] = self.results[metric.name]
        report = pd.DataFrame(data)
        report.index.rename("Iteration", inplace=True)
        report.name = self.dataset_name
        report = report.round(4)
        self._dataframe = report
        return report

    def __str__(self):
        """Create and return a tabular report of the evaluation."""
        return self.as_dataframe().to_string()


def evaluate(
    dataset: Dataset,
    ignore_node_features: bool = False,
    ignore_edge_features: bool = True,
    additional_features: List[str] = None,
    metrics: Union[List[str], _Metric] = None,
    iterations: int = 3,
) -> EvaluationResult:
    """Evaluate a dataset.

    This method analyzes how "hard" a dataset is to classify from 1-WL perspective.
    It does so by computing the following metrics:
        1. 1-WL-identifiability: the percentage of graphs that can be uniquely
            identified with 1-WL algorithm at k iterations. Calculated if
            len(dataset) > 0.
        2. Upper bound accuracy: the upper bound accuracy of 1-WL algorithm at
            k iterations for labeled datasets, considering majority vote. Calculated
            if len(dataset) > 0.

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
            - "upper_bound_accuracy_node": Upper bound accuracy (node labels)
    iterations : int, optional
        The number of iterations to run 1-WL for, by default 3

    Returns
    -------
    result : EvaluationResult
        The evaluation results object.
    """
    metrics: list[_Metric] = _init_metrics(dataset, metrics)

    graphs = dataset.graphs
    if ignore_node_features or ignore_edge_features:
        graphs = _clean_graphs(ignore_node_features, ignore_edge_features, graphs)
    if additional_features is not None:
        graphs = [label(graph, methods=additional_features) for graph in graphs]

    labels = dataset.labels
    node_labels = dataset.node_labels
    edge_labels = dataset.edge_labels

    estimate_edge_hashes = any(metric.type == "edge" for metric in metrics)
    estimate_node_hashes = estimate_edge_hashes or any(
        metric.type == "node" or metric.type == "graph_node" for metric in metrics
    )
    hashes, node_hashes = _estimate_hashes_at_k_iterations(
        graphs.copy(), iterations, estimate_node_hashes
    )

    edge_hashes = None
    if estimate_edge_hashes:
        edge_hashes = _estimate_edge_hashes(graphs, node_hashes)

    results = {}
    for metric in metrics:
        results[metric.name] = metric.evaluate(
            graphs, labels, node_labels, edge_labels, hashes, node_hashes, edge_hashes
        )

    result = EvaluationResult(dataset, metrics, results)

    return result


def _clean_graphs(ignore_node_features, ignore_edge_features, graphs):
    return [
        ig.Graph(
            n=graph.vcount(),
            edges=graph.get_edgelist(),
            directed=graph.is_directed(),
            vertex_attrs={attr: graph.vs[attr] for attr in graph.vertex_attributes()}
            if not ignore_node_features
            else {},
            edge_attrs={attr: graph.es[attr] for attr in graph.edge_attributes()}
            if not ignore_edge_features
            else {},
        )
        for graph in graphs
    ]


def _init_metrics(dataset, metrics):
    # if list of _Metric objects is passed, use it
    if metrics is not None and isinstance(metrics[0], _Metric):
        return metrics
    if metrics is None:
        metrics = []
        if len(dataset) > 0:
            metrics.append(DEFAULT_METRICS["identifiability"])
        if dataset.labels is not None:
            metrics.append(DEFAULT_METRICS["upper_bound_accuracy"])
        if dataset.node_labels is not None:
            metrics.append(DEFAULT_METRICS["upper_bound_accuracy_node"])
        if dataset.edge_labels is not None:
            metrics.append(DEFAULT_METRICS["upper_bound_accuracy_edge"])
    else:
        metrics = [DEFAULT_METRICS[metric] for metric in metrics]
    return metrics


def _estimate_hashes_at_k_iterations(
    graphs: List[ig.Graph],
    iterations: int = 3,
    return_node_hashes: bool = False,
) -> Tuple[Dict[int, List[str]], Dict[int, List[List[str]]]]:
    """Estimate the 1-WL hashes of a dataset at different iterations.

    We simply run 1-WL on all graphs for k iterations and count the
    number of graphs that have a unique 1-WL representation at each iteration.

    Parameters
    ----------
    graphs : List[ig.Graph]
        The graphs to estimate the hashes of.
    iterations : int, optional
        The number of iterations to run 1-WL for, by default 3
    return_node_hashes : bool, optional
        Whether to return the estimated node hashes, by default False.
        If False, second argument of the returned tuple is None.

    Returns
    -------
    hashes : dict[int, List[str]]
        The estimated hashes of the graphs.
    node_hashes : dict[int, List[List[str]]]
        The estimated node hashes of the graphs. Only returned if
        `return_node_hashes` is True.
    """
    k = 0
    hashes = {}
    node_hashes = {} if return_node_hashes else None
    stabilized_graphs = set()
    graph_count = len(graphs)
    last_graph_refinements = graphs
    last_graph_hashes = [None] * graph_count
    last_node_hashes = [None] * graph_count if return_node_hashes else None
    while k <= iterations:
        new_hashes, new_node_hashes, new_graph_refinements = (
            last_graph_hashes,
            last_node_hashes,
            last_graph_refinements,
        )
        for idx, graph in enumerate(last_graph_refinements):
            if idx in stabilized_graphs:
                continue
            edge_attrs = graph.es.attributes()
            node_attrs = graph.vs.attributes() if k == 0 else "label"
            # zeroth iteration is just the original node labels
            iter = 0 if k == 0 else 1
            # Do a single iteration of 1-WL
            graph_hash, refined_graph = weisfeiler_lehman_hash(
                graph,
                edge_attrs,
                node_attrs,
                iterations=iter,
                return_graph=True,
            )
            # Check if 1-WL has stabilized
            if k > 2 and graph_hash == last_graph_hashes[idx]:
                stabilized_graphs.add(idx)
            new_graph_refinements[idx] = refined_graph
            new_hashes[idx] = graph_hash
            if return_node_hashes:
                new_node_hashes[idx] = refined_graph.vs["label"]
        hashes[k] = new_hashes.copy()
        if return_node_hashes:
            node_hashes[k] = new_node_hashes.copy()
        last_graph_refinements = new_graph_refinements
        last_graph_hashes = new_hashes
        last_node_hashes = new_node_hashes
        k += 1
    return (hashes, node_hashes)


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
    hashes: Dict[int, List[str]],
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


def _estimate_edge_hashes(
    graphs: List[ig.Graph],
    node_hashes: Dict[int, List[List[str]]],
) -> Dict[int, List[List[str]]]:
    """Estimate the edge hashes of a dataset.

    For each edge, edge hash is the hashes of the nodes at the ends of the edge.
    Which is simply sorted concatenation of the node hashes.

    Parameters
    ----------
    graphs : List[ig.Graph]
        The graphs in the dataset.
    node_hashes : dict[int, List[List[str]]]
        The estimated hashes of the nodes at different iterations.

    Returns
    -------
    edge_hashes : dict[int, List[List[str]]]
        The estimated hashes of the edges at different iterations.
    """
    edge_hashes = {}
    for k, hashes_at_k in node_hashes.items():
        edge_hashes[k] = []
        for graph, node_hashes_at_k in zip(graphs, hashes_at_k):
            edge_hashes_at_k = []
            for edge in graph.es:
                edge_hashes_at_k.append(
                    ":".join(
                        sorted(
                            [
                                node_hashes_at_k[edge.source],
                                node_hashes_at_k[edge.target],
                            ]
                        )
                    )
                )
            edge_hashes[k].append(edge_hashes_at_k)
    return edge_hashes


def _evaluate_upper_bound_accuracy(
    hashes: Dict[int, List[str]],
    labels: List[int],
) -> dict:
    """Evaluate the upper bound accuracy of a dataset.

    This algorithm estimates the upper bound accuracy of a dataset by first
    computing the majority vote of the labels of the graphs that have the same
    1-WL representation at different iterations. For these majority labels,
    the accuracy is then computed.

    Parameters
    ----------
    hashes : dict[int, List[str]]
        The estimated hashes of the graphs/nodes/edges at different iterations.
    labels : List[int]
        The labels of the graphs/nodes/edges in the dataset.

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


def _evaluate_mse(
    hashes: Dict[int, List[str]],
    labels: List[float],
) -> dict:
    """Evaluate the mean squared error of a dataset for given node labels.

    This algorithm estimates the mean squared error of a dataset by first
    computing the average of the labels of the nodes that have the same
    1-WL representation at different iterations. For these average labels,
    the mean squared error is then computed.

    Parameters
    ----------
    hashes : dict[int, List[str]]
        The estimated hashes of the graphs/nodes/edges at different iterations.
    labels : List[float]
        The labels of the graphs/nodes/edges in the dataset.

    Returns
    -------
    mse : dict[int, float]
        The mean squared error of the dataset.
    """
    mse = {}
    for k, hashes_at_k in hashes.items():
        average_labels = {}
        for hash, lbl in zip(hashes_at_k, labels):
            if hash not in average_labels:
                average_labels[hash] = []
            average_labels[hash].append(lbl)
        average_labels = {
            hash: sum(labels) / len(labels) for hash, labels in average_labels.items()
        }
        predicted_labels = [average_labels[hash] for hash in hashes_at_k]
        mse[k] = sum(
            [(y_true - y_pred) ** 2 for y_true, y_pred in zip(labels, predicted_labels)]
        ) / len(labels)
    return mse


def _evaluate_f1(
    hashes: Dict[int, List[List[str]]],
    labels: List[List[float]],
) -> dict:
    """Evaluate the F1 score of a dataset for given node labels.

    This algorithm estimates the F1 score of a dataset by first
    computing the majority vote of the labels of the nodes that have the same
    1-WL representation at different iterations. For these majority labels,
    the F1 score is then computed.

    Parameters
    ----------
    hashes : dict[int, List[str]]
        The estimated hashes of the graphs/nodes/edges at different iterations.
    labels : List[int]
        The labels of the graphs/nodes/edges in the dataset.

    Returns
    -------
    f1 : dict[int, float]
        The F1 score of the dataset.
    """
    f1 = {}
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
        precision = sum(
            [y_true == y_pred for y_true, y_pred in zip(labels, predicted_labels)]
        ) / len(labels)
        recall = sum(
            [y_true == y_pred for y_true, y_pred in zip(labels, predicted_labels)]
        ) / len(labels)
        f1[k] = 2 * (precision * recall) / (precision + recall)
    return f1

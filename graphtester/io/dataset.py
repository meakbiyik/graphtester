"""Dataset class to manage graphs and optional labels."""

import lzma
import pickle
from typing import List, Tuple

import igraph as ig
import numpy as np


class Dataset:
    """Dataset class to manage graphs and optional labels.

    This class is not meant to be created directly. Instead, use the
    `load` function to load a dataset from an arbitrary source.
    """

    def __init__(
        self,
        graphs: List[ig.Graph],
        labels: List[float] = None,
        node_labels: List[List[float]] = None,
        edge_labels: List[List[float]] = None,
        name: str = None,
    ):
        """Initialize a Dataset object.

        Parameters
        ----------
        graphs : List[ig.Graph]
            The graphs.
        labels : List[float], optional
            The labels of the graphs. If None (default), the graphs are
            assumed to be unlabeled.
        node_labels : List[List[float]], optional
            The labels of the nodes. If None (default), the nodes
            are assumed to be unlabeled. Used for node classification tasks.
        edge_labels : List[List[float]], optional
            The labels of the edges. If None (default), the edges
            are assumed to be unlabeled. Used for edge classification tasks.
        name : str, optional
            The name of the dataset.
        """
        self.graphs = graphs
        self.labels = labels
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.name = name if name is not None else "Unnamed Dataset"

    @classmethod
    def from_dgl(
        cls,
        dgl_dataset,
        labels: List[float] = None,
        node_labels: List[List[float]] = None,
        edge_labels: List[List[float]] = None,
    ) -> "Dataset":
        """Create a Dataset from a DGL dataset.

        Parameters
        ----------
        dgl_dataset : DGLDataset
            The DGL dataset.
        labels : List[float], optional
            The labels of the graphs. If None (default), the labels in the
            dataset are used, if available. If the dataset is already labeled,
            the labels override the labels in the dataset.
        node_labels : List[List[float]], optional
            The labels of the nodes. If None (default), the node labels
            in the dataset are used, if available. If the dataset is already
            labeled, the node labels override the node labels in the dataset.
        edge_labels : List[List[float]], optional
            The labels of the edges. If None (default), the edge labels
            in the dataset are used, if available. If the dataset is already
            labeled, the edge labels override the edge labels in the dataset.

        Returns
        -------
        dataset : Dataset
            The Dataset.
        """
        import dgl.backend as F

        with_graph_labels = isinstance(dgl_dataset[0], tuple)
        first_graph = dgl_dataset[0][0] if with_graph_labels else dgl_dataset[0]
        with_node_labels = "label" in first_graph.ndata
        with_edge_labels = (
            ("e_type" in first_graph.edata)
            or ("rel_type" in first_graph.edata)
            or ("etype" in first_graph.edata)
        )
        node_attr = list(first_graph.ndata.keys())
        edge_attr = list(first_graph.edata.keys())

        edge_label_attr = None
        if with_edge_labels:
            edge_label_attr = (
                "e_type"
                if "e_type" in first_graph.edata
                else "rel_type"
                if "rel_type" in first_graph.edata
                else "etype"
            )

        graph_count = len(dgl_dataset)
        gget = (
            (lambda i: dgl_dataset[i][0])
            if with_graph_labels
            else dgl_dataset.__getitem__
        )
        lget = (
            (lambda i: float(dgl_dataset[i][1]))
            if with_graph_labels
            else lambda _: None
        )
        nlget = (
            (lambda i: [float(lbl) for lbl in gget(i).ndata["label"].tolist()])
            if with_node_labels
            else lambda _: None
        )
        elget = (
            (lambda i: [float(lbl) for lbl in gget(i).edata[edge_label_attr].tolist()])
            if with_edge_labels
            else lambda _: None
        )
        graphs, _labels, _node_labels, _edge_labels = zip(
            *[
                (
                    ig.Graph.from_networkx(gget(i).to_networkx(node_attr, edge_attr)),
                    lget(i),
                    nlget(i),
                    elget(i),
                )
                for i in range(graph_count)
            ]
        )
        if labels is None and with_graph_labels:
            labels = list(_labels)
        if node_labels is None and with_node_labels:
            node_labels = list(_node_labels)
        if edge_labels is None and with_edge_labels:
            edge_labels = list(_edge_labels)

        # Remove superfluous attributes and convert tensors to lists
        # Also remove the node_label attribute if exists
        attrs_to_remove = ["_ID", "id", "_nx_name", "label", edge_label_attr]
        for graph in graphs:
            for attr in attrs_to_remove:
                if attr in graph.vs.attributes():
                    del graph.vs[attr]
            for attr in attrs_to_remove:
                if attr in graph.es.attributes():
                    del graph.es[attr]
            for attr in graph.vs.attributes():
                graph.vs[attr] = [
                    np.squeeze(F.asnumpy(x)).tolist() for x in graph.vs[attr]
                ]
            for attr in graph.es.attributes():
                graph.es[attr] = [
                    np.squeeze(F.asnumpy(x)).tolist() for x in graph.es[attr]
                ]

        return cls(
            list(graphs),
            labels,
            node_labels,
            edge_labels,
            dgl_dataset.name,
        )

    @classmethod
    def from_pickle(cls, path: str) -> "Dataset":
        """Load a Dataset from a compressed pickle file.

        Parameters
        ----------
        path : str
            The path to the pickle file.

        Returns
        -------
        dataset : Dataset
            The Dataset.
        """
        with lzma.open(path, "rb") as file:
            return pickle.load(file)

    def to_pickle(self, path: str):
        """Save the dataset to a compressed pickle file.

        Parameters
        ----------
        path : str
            The path to the pickle file.
        """
        with lzma.open(path, "wb") as file:
            pickle.dump(self, file)

    @property
    def is_labeled(self) -> bool:
        """Check if the dataset is labeled.

        Returns
        -------
        is_labeled : bool
            Whether the dataset is labeled.
        """
        return self.labels is not None

    def __len__(self) -> int:
        """Get the number of graphs in the dataset.

        Returns
        -------
        length : int
            The number of graphs in the dataset.
        """
        return len(self.graphs)

    def __getitem__(self, index: int) -> Tuple[ig.Graph, int]:
        """Get a graph and its label.

        Parameters
        ----------
        index : int
            The index of the graph.

        Returns
        -------
        graph : ig.Graph
            The graph.
        label : int
            The label of the graph.
        """
        return self.graphs[index], self.labels[index]

    def __iter__(self) -> Tuple[ig.Graph, float]:
        """Iterate over the graphs and their labels.

        Yields
        ------
        graph : ig.Graph
            The graph.
        label : float
            The label of the graph.
        """
        for graph, label in zip(self.graphs, self.labels):
            yield graph, label

    def __repr__(self) -> str:
        """Get a string representation of the dataset.

        Returns
        -------
        repr : str
            A string representation of the dataset.
        """
        return (
            f"Dataset(name={self.name}, "
            f"graph_count={len(self)}, "
            f"is_labeled={self.is_labeled})"
        )

    def __str__(self) -> str:
        """Get a string representation of the dataset.

        Returns
        -------
        repr : str
            A string representation of the dataset.
        """
        return self.__repr__()

    def __add__(self, other: "Dataset") -> "Dataset":
        """Add two datasets.

        Parameters
        ----------
        other : Dataset
            The other dataset.

        Returns
        -------
        dataset : Dataset
            The combined dataset.
        """
        return Dataset(
            graphs=self.graphs + other.graphs,
            labels=self.labels + other.labels,
        )

    def __contains__(self, graph: ig.Graph) -> bool:
        """Check if a graph is in the dataset.

        Parameters
        ----------
        graph : ig.Graph
            The graph to check.

        Returns
        -------
        contains : bool
            Whether the graph is in the dataset.
        """
        return graph in self.graphs

    def __eq__(self, other: "Dataset") -> bool:
        """Check if two datasets are equal.

        Parameters
        ----------
        other : Dataset
            The other dataset.

        Returns
        -------
        equal : bool
            Whether the datasets are equal.
        """
        return self.graphs == other.graphs and self.labels == other.labels

    def __ne__(self, other: "Dataset") -> bool:
        """Check if two datasets are not equal.

        Parameters
        ----------
        other : Dataset
            The other dataset.

        Returns
        -------
        not_equal : bool
            Whether the datasets are not equal.
        """
        return not self.__eq__(other)

    def copy(self) -> "Dataset":
        """Get a copy of the dataset.

        Returns
        -------
        copy : Dataset
            A copy of the dataset.
        """
        return Dataset(
            graphs=[graph.copy() for graph in self.graphs],
            labels=self.labels.copy() if self.labels is not None else None,
        )

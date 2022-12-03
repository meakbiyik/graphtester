"""Common loader API to procure graphs.

The users can either load a synthetic dataset named and versioned by graphtester,
or load a real dataset. Graphtester provides some real datasets that can be loaded
by their names similar to the synthetic datasets. Alternatively, the user can
provide a list of networkx or igraph graphs to be used for testing.
"""

from pathlib import Path
from typing import List, Tuple

import igraph as ig
import networkx as nx

from graphtester.io.dataset import Dataset

DATASETS = {
    "GT": Path(__file__).parent / "datasets" / "GT.pkl",
    "GT-small": Path(__file__).parent / "datasets" / "GT-small.pkl",
    "MUTAG": lambda dgl: dgl.data.TUDataset("MUTAG"),
    "ENZYMES": lambda dgl: dgl.data.TUDataset("ENZYMES"),
    "DD": lambda dgl: dgl.data.TUDataset("DD"),
    "COLLAB": lambda dgl: dgl.data.TUDataset("COLLAB"),
    "PROTEINS": lambda dgl: dgl.data.TUDataset("PROTEINS"),
}


def load(
    name_or_graphs: str | list, classes: List = None, dataset_name: str = None
) -> Dataset:
    """Load graphs from a dataset or a list of graphs.

    Parameters
    ----------
    name_or_graphs : str or List[Union[nx.Graph, ig.Graph]]
        The name of the dataset to load, or a list of graphs.
    classes : List[any], optional
        The classes of the graphs for classification tasks. If None (default),
        the graphs are assumed to be unlabeled. Needs to be the same length as
        the number of graphs.

    Returns
    -------
    dataset : Dataset
    """
    if isinstance(name_or_graphs, str):
        dataset = _load_dataset(name_or_graphs, classes)
    else:
        dataset = _load_graphs(name_or_graphs, classes)

    if dataset_name is not None:
        dataset.name = dataset_name

    return dataset


def _load_dataset(name: str, classes: List = None) -> Tuple[List, List]:
    """Load a dataset.

    Parameters
    ----------
    name : str
        The name of the dataset to load.
    classes : List[any], optional
        The classes of the graphs for classification tasks. If None (default),
        the labels in the dataset are used, if available. If the dataset is
        already labeled, the classes override the labels.

    Returns
    -------
    dataset : Dataset
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}.")

    if name.startswith("GT"):
        return Dataset.from_pickle(DATASETS[name])

    # Otherwise a DGL dataset
    dgl = _import_dgl()
    dgl_dataset = DATASETS[name](dgl)
    return Dataset.from_dgl(dgl_dataset)


def _import_dgl():
    """Import dgl."""
    try:
        import dgl
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "DGL is used for named datasets, which requires a backend. "
            "Please install one of PyTorch, MXNet, or TensorFlow."
        )
    return dgl


def _load_graphs(graphs: List[nx.Graph | ig.Graph], classes: List = None) -> Dataset:
    """Load graphs.

    Parameters
    ----------
    graphs : List[Union[nx.Graph, ig.Graph]]
        The graphs.
    classes : List[any], optional
        The classes of the graphs for classification tasks. If None (default),
        the graphs are assumed to be unlabeled.

    Returns
    -------
    dataset : Dataset
    """
    if not isinstance(graphs, list):
        raise TypeError("The graphs must be a list.")

    if not isinstance(graphs[0], (nx.Graph, ig.Graph)):
        raise TypeError("The graphs must be a list of networkx or igraph graphs.")

    if isinstance(graphs[0], nx.Graph):
        graphs = [ig.Graph.from_networkx(g) for g in graphs]

    return Dataset(graphs, classes)

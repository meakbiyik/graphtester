"""Common loader API to procure graphs.

The users can either load a synthetic dataset named and versioned by graphtester,
or load a real dataset. Graphtester provides some real datasets that can be loaded
by their names similar to the synthetic datasets. Alternatively, the user can
provide a list of networkx or igraph graphs to be used for testing.
"""

from pathlib import Path
from typing import List

import igraph as ig
import networkx as nx

from graphtester.io.dataset import Dataset

DATASETS = {
    # Homebrewed datasets
    "GT": Path(__file__).parent / "datasets" / "GT.pkl",
    "GT-small": Path(__file__).parent / "datasets" / "GT-small.pkl",
    # TU datasets
    # Small molecules - TU
    "AIDS": lambda dgl: dgl.data.TUDataset("AIDS"),
    "BZR": lambda dgl: dgl.data.TUDataset("BZR"),
    "BZR_MD": lambda dgl: dgl.data.TUDataset("BZR_MD"),
    "COX2": lambda dgl: dgl.data.TUDataset("COX2"),
    "COX2_MD": lambda dgl: dgl.data.TUDataset("COX2_MD"),
    "DHFR": lambda dgl: dgl.data.TUDataset("DHFR"),
    "DHFR_MD": lambda dgl: dgl.data.TUDataset("DHFR_MD"),
    "ER_MD": lambda dgl: dgl.data.TUDataset("ER_MD"),
    "FRANKENSTEIN": lambda dgl: dgl.data.TUDataset("FRANKENSTEIN"),
    "Mutagenicity": lambda dgl: dgl.data.TUDataset("Mutagenicity"),
    "MUTAG": lambda dgl: dgl.data.TUDataset("MUTAG"),
    "NCI1": lambda dgl: dgl.data.TUDataset("NCI1"),
    "NCI109": lambda dgl: dgl.data.TUDataset("NCI109"),
    "PTC_FM": lambda dgl: dgl.data.TUDataset("PTC_FM"),
    "PTC_FR": lambda dgl: dgl.data.TUDataset("PTC_FR"),
    "PTC_MM": lambda dgl: dgl.data.TUDataset("PTC_MM"),
    "PTC_MR": lambda dgl: dgl.data.TUDataset("PTC_MR"),
    # Bioinformatics - TU
    "ENZYMES": lambda dgl: dgl.data.TUDataset("ENZYMES"),
    "DD": lambda dgl: dgl.data.TUDataset("DD"),
    "PROTEINS": lambda dgl: dgl.data.TUDataset("PROTEINS"),
    # Computer vision - TU
    "Fingerprint": lambda dgl: dgl.data.TUDataset("Fingerprint"),
    "Cuneiform": lambda dgl: dgl.data.TUDataset("Cuneiform"),
    "COIL-DEL": lambda dgl: dgl.data.TUDataset("COIL-DEL"),
    "COIL-RAG": lambda dgl: dgl.data.TUDataset("COIL-RAG"),
    "MSRC_9": lambda dgl: dgl.data.TUDataset("MSRC_9"),
    # Social networks - TU
    "IMDB-BINARY": lambda dgl: dgl.data.TUDataset("IMDB-BINARY"),
    "IMDB-MULTI": lambda dgl: dgl.data.TUDataset("IMDB-MULTI"),
    "COLLAB": lambda dgl: dgl.data.TUDataset("COLLAB"),
    "REDDIT-BINARY": lambda dgl: dgl.data.TUDataset("REDDIT-BINARY"),
    "REDDIT-MULTI-5K": lambda dgl: dgl.data.TUDataset("REDDIT-MULTI-5K"),
    # GIN datasets
    "PTC": lambda dgl: dgl.data.GINDataset("PTC", False),
    # OGB datasets
    # graph classification
    "ogbg-molbace": lambda ogbg: ogbg.DglGraphPropPredDataset("ogbg-molbace"),
    "ogbg-molbbbp": lambda ogbg: ogbg.DglGraphPropPredDataset("ogbg-molbbbp"),
    "ogbg-molclintox": lambda ogbg: ogbg.DglGraphPropPredDataset("ogbg-molclintox"),
    "ogbg-molmuv": lambda ogbg: ogbg.DglGraphPropPredDataset("ogbg-molmuv"),
    "ogbg-molsider": lambda ogbg: ogbg.DglGraphPropPredDataset("ogbg-molsider"),
    "ogbg-molhiv": lambda ogbg: ogbg.DglGraphPropPredDataset("ogbg-molhiv"),
    "ogbg-moltox21": lambda ogbg: ogbg.DglGraphPropPredDataset("ogbg-moltox21"),
    "ogbg-moltoxcast": lambda ogbg: ogbg.DglGraphPropPredDataset("ogbg-moltoxcast"),
    # node classification
    "ogbn-proteins": lambda ogbn: ogbn.DglNodePropPredDataset("ogbn-proteins"),
    # DGL node classification datasets
    "Cora": lambda dgl: dgl.data.CoraGraphDataset(),
    "Citeseer": lambda dgl: dgl.data.CiteseerGraphDataset(),
}


def load(
    name_or_graphs: "str | list | dgl.data.DGLDataset",  # type: ignore # noqa: F821
    labels: list[int] = None,
    node_labels: list[list[int]] = None,
    dataset_name: str = None,
) -> Dataset:
    """Load graphs from a dataset or a list of graphs.

    Parameters
    ----------
    name_or_graphs : str or List[Union[nx.Graph, ig.Graph, dgl.data.DGLDataset]]
        The name of the dataset to load, or a list of graphs.
    labels : List[int], optional
        The labels of the graphs for classification tasks. If None (default),
        the graphs are assumed to be unlabeled. Needs to be the same length as
        the number of graphs.
    node_labels : List[List[int]], optional
        The node labels of the graphs for node classification tasks. If None
        (default), the graphs are assumed to be unlabeled. Needs to be the same
        length as the number of graphs.
    dataset_name : str, optional
        The name of the dataset. If None (default), the name of the dataset is
        used.

    Returns
    -------
    dataset : Dataset
    """
    if isinstance(name_or_graphs, str):
        dataset = _load_dataset(name_or_graphs, labels, node_labels)
    elif isinstance(name_or_graphs, list):
        dataset = _load_graphs(name_or_graphs, labels, node_labels)
    else:
        dataset = Dataset.from_dgl(name_or_graphs, labels, node_labels)

    if dataset_name is not None:
        dataset.name = dataset_name

    return dataset


def _load_dataset(name: str, labels: List = None, node_labels: List = None) -> Dataset:
    """Load a dataset.

    Parameters
    ----------
    name : str
        The name of the dataset to load.
    labels : List[any], optional
        The labels of the graphs for classification tasks. If None (default),
        the labels in the dataset are used, if available. If the dataset is
        already labeled, the labels override the labels.
    node_labels : List[List[any]], optional
        The node labels of the graphs for node classification tasks. If None
        (default), the node labels in the dataset are used, if available. If the
        dataset is already labeled, the node labels override the node labels.

    Returns
    -------
    dataset : Dataset
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}.")

    if name.startswith("GT"):
        return Dataset.from_pickle(DATASETS[name])

    if name.startswith("ogbg"):
        ogb = _import_ogbg()
        ogb_dataset = DATASETS[name](ogb)
        return Dataset.from_dgl(ogb_dataset, labels, node_labels)

    if name.startswith("ogbn"):
        ogb = _import_ogbn()
        ogb_dataset = DATASETS[name](ogb)
        return Dataset.from_dgl(ogb_dataset, labels, node_labels)

    # Otherwise a DGL dataset
    dgl = _import_dgl()
    dgl_dataset = DATASETS[name](dgl)
    return Dataset.from_dgl(dgl_dataset, labels, node_labels)


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


def _import_ogbg():
    """Import ogb.graphpropped."""
    try:
        import ogb.graphproppred as ogbg
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "ogb is used for OGB datasets, which requires" "PyTorch to be installed."
        )
    return ogbg


def _import_ogbn():
    """Import ogb.nodepropped."""
    try:
        import ogb.nodeproppred as ogbn
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "ogb is used for OGB datasets, which requires" "PyTorch to be installed."
        )
    return ogbn


def _load_graphs(
    graphs: List[nx.Graph | ig.Graph], labels: List = None, node_labels: List = None
) -> Dataset:
    """Load graphs.

    Parameters
    ----------
    graphs : List[Union[nx.Graph, ig.Graph]]
        The graphs.
    labels : List[any], optional
        The labels of the graphs for classification tasks. If None (default),
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

    return Dataset(graphs, labels, node_labels)

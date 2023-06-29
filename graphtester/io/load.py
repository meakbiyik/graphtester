"""Common loader API to procure graphs.

The users can either load a synthetic dataset named and versioned by graphtester,
or load a real dataset. Graphtester provides some real datasets that can be loaded
by their names similar to the synthetic datasets. Alternatively, the user can
provide a list of networkx or igraph graphs to be used for testing.
"""
import os
from pathlib import Path
from typing import List, Union

import igraph as ig
import networkx as nx

from graphtester.io.dataset import Dataset

if os.getenv("GRAPHTESTER_CACHE_DIR") is not None:
    GRAPHTESTER_CACHE_DIR = Path(os.getenv("GRAPHTESTER_CACHE_DIR"))
    GRAPHTESTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dgl_param = {"raw_dir": str(GRAPHTESTER_CACHE_DIR)}
    ogb_param = {"root": str(GRAPHTESTER_CACHE_DIR)}
    pyg_param = {"root": str(GRAPHTESTER_CACHE_DIR)}
else:
    GRAPHTESTER_CACHE_DIR = None
    cache_dir = Path(__file__).parent / ".gtcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dgl_param = {"raw_dir": str(cache_dir)}
    ogb_param = {"root": str(cache_dir)}
    pyg_param = {"root": str(cache_dir)}

DATASETS = {
    # Homebrewed datasets
    "GT": Path(__file__).parent / "datasets" / "GT.pkl",
    "GT-small": Path(__file__).parent / "datasets" / "GT-small.pkl",
    # PyTorch geometric datasets
    "ZINC_FULL": lambda pyg: pyg.datasets.ZINC(**pyg_param),
    "ZINC": lambda pyg: pyg.datasets.ZINC(subset=True, **pyg_param),
    "MNIST": lambda pyg: pyg.datasets.GNNBenchmarkDataset(name="MNIST", **pyg_param),
    "CIFAR10": lambda pyg: pyg.datasets.GNNBenchmarkDataset(
        name="CIFAR10", **pyg_param
    ),
    "PATTERN": lambda pyg: pyg.datasets.GNNBenchmarkDataset(
        name="PATTERN", **pyg_param
    ),
    "CLUSTER": lambda pyg: pyg.datasets.GNNBenchmarkDataset(
        name="CLUSTER", **pyg_param
    ),
    # TU datasets
    # Small molecules - TU
    "AIDS": lambda dgl: dgl.data.TUDataset("AIDS", **dgl_param),
    "BZR": lambda dgl: dgl.data.TUDataset("BZR", **dgl_param),
    "BZR_MD": lambda dgl: dgl.data.TUDataset("BZR_MD", **dgl_param),
    "COX2": lambda dgl: dgl.data.TUDataset("COX2", **dgl_param),
    "COX2_MD": lambda dgl: dgl.data.TUDataset("COX2_MD", **dgl_param),
    "DHFR": lambda dgl: dgl.data.TUDataset("DHFR", **dgl_param),
    "DHFR_MD": lambda dgl: dgl.data.TUDataset("DHFR_MD", **dgl_param),
    "ER_MD": lambda dgl: dgl.data.TUDataset("ER_MD", **dgl_param),
    "FRANKENSTEIN": lambda dgl: dgl.data.TUDataset("FRANKENSTEIN", **dgl_param),
    "Mutagenicity": lambda dgl: dgl.data.TUDataset("Mutagenicity", **dgl_param),
    "MUTAG": lambda dgl: dgl.data.TUDataset("MUTAG", **dgl_param),
    "NCI1": lambda dgl: dgl.data.TUDataset("NCI1", **dgl_param),
    "NCI109": lambda dgl: dgl.data.TUDataset("NCI109", **dgl_param),
    "PTC_FM": lambda dgl: dgl.data.TUDataset("PTC_FM", **dgl_param),
    "PTC_FR": lambda dgl: dgl.data.TUDataset("PTC_FR", **dgl_param),
    "PTC_MM": lambda dgl: dgl.data.TUDataset("PTC_MM", **dgl_param),
    "PTC_MR": lambda dgl: dgl.data.TUDataset("PTC_MR", **dgl_param),
    # Bioinformatics - TU
    "ENZYMES": lambda dgl: dgl.data.TUDataset("ENZYMES", **dgl_param),
    "DD": lambda dgl: dgl.data.TUDataset("DD", **dgl_param),
    "PROTEINS": lambda dgl: dgl.data.TUDataset("PROTEINS", **dgl_param),
    # Computer vision - TU
    "Fingerprint": lambda dgl: dgl.data.TUDataset("Fingerprint", **dgl_param),
    "Cuneiform": lambda dgl: dgl.data.TUDataset("Cuneiform", **dgl_param),
    "COIL-DEL": lambda dgl: dgl.data.TUDataset("COIL-DEL", **dgl_param),
    "COIL-RAG": lambda dgl: dgl.data.TUDataset("COIL-RAG", **dgl_param),
    "MSRC_9": lambda dgl: dgl.data.TUDataset("MSRC_9", **dgl_param),
    # Social networks - TU
    "IMDB-BINARY": lambda dgl: dgl.data.TUDataset("IMDB-BINARY", **dgl_param),
    "IMDB-MULTI": lambda dgl: dgl.data.TUDataset("IMDB-MULTI", **dgl_param),
    "COLLAB": lambda dgl: dgl.data.TUDataset("COLLAB", **dgl_param),
    "REDDIT-BINARY": lambda dgl: dgl.data.TUDataset("REDDIT-BINARY", **dgl_param),
    "REDDIT-MULTI-5K": lambda dgl: dgl.data.TUDataset("REDDIT-MULTI-5K", **dgl_param),
    # OGB datasets
    # graph classification
    "ogbg-molbace": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-molbace", **ogb_param
    ),
    "ogbg-molbbbp": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-molbbbp", **ogb_param
    ),
    "ogbg-molclintox": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-molclintox", **ogb_param
    ),
    "ogbg-molmuv": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-molmuv", **ogb_param
    ),
    "ogbg-molsider": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-molsider", **ogb_param
    ),
    "ogbg-molhiv": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-molhiv", **ogb_param
    ),
    "ogbg-molpcba": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-molpcba", **ogb_param
    ),
    "ogbg-moltox21": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-moltox21", **ogb_param
    ),
    "ogbg-moltoxcast": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-moltoxcast", **ogb_param
    ),
    # node classification
    "ogbn-arxiv": lambda ogbn: ogbn.DglNodePropPredDataset("ogbn-arxiv", **ogb_param),
    "ogbn-proteins": lambda ogbn: ogbn.DglNodePropPredDataset(
        "ogbn-proteins", **ogb_param
    ),
    "ogbn-products": lambda ogbn: ogbn.DglNodePropPredDataset(
        "ogbn-products", **ogb_param
    ),
    # link prediction
    # "ogbl-collab": lambda ogbl: ogbl.DglLinkPropPredDataset("ogbl-collab", **ogb_param), # noqa: E501
    # "ogbl-ddi": lambda ogbl: ogbl.DglLinkPropPredDataset("ogbl-ddi", **ogb_param),
    # "ogbl-ppa": lambda ogbl: ogbl.DglLinkPropPredDataset("ogbl-ppa", **ogb_param),
    # "ogbl-citation2": lambda ogbl: ogbl.DglLinkPropPredDataset("ogbl-citation2", **ogb_param), # noqa: E501
    # graph regression
    "ogbg-molesol": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-molesol", **ogb_param
    ),
    "ogbg-molfreesolv": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-molfreesolv", **ogb_param
    ),
    "ogbg-mollipo": lambda ogbg: ogbg.DglGraphPropPredDataset(
        "ogbg-mollipo", **ogb_param
    ),
    # DGL node classification datasets
    "Cora": lambda dgl: dgl.data.CoraGraphDataset(**dgl_param),
    "Citeseer": lambda dgl: dgl.data.CiteseerGraphDataset(**dgl_param),
    "Pubmed": lambda dgl: dgl.data.PubmedGraphDataset(**dgl_param),
    "CoauthorCS": lambda dgl: dgl.data.CoauthorCSDataset(**dgl_param),
    "CoauthorPhysics": lambda dgl: dgl.data.CoauthorPhysicsDataset(**dgl_param),
    "AmazonCoBuyComputer": lambda dgl: dgl.data.AmazonCoBuyComputerDataset(**dgl_param),
    # DGL link prediction datasets
    # "FB15k237": lambda dgl: dgl.data.FB15k237Dataset(**dgl_param),
    # "WN18Dataset": lambda dgl: dgl.data.WN18Dataset(**dgl_param),
}


def load(
    name_or_graphs: "str | list | dgl.data.DGLDataset | torch_geometric.data.Dataset",  # type: ignore # noqa: F821, E501
    labels: List[float] = None,
    node_labels: List[List[float]] = None,
    edge_labels: List[List[float]] = None,
    dataset_name: str = None,
    graph_count: int = None,
    seed: int = 0,
) -> Dataset:
    """Load graphs from a dataset or a list of graphs.

    Parameters
    ----------
    name_or_graphs : str or List[Union[nx.Graph, ig.Graph, dgl.data.DGLDataset, torch_geometric.data.Dataset]] # noqa: E501
        The name of the dataset to load, or a list of graphs, or a dataset
        object from DGL or PyTorch Geometric.
    labels : List[float], optional
        The labels of the graphs for classification tasks. If None (default),
        the graphs are assumed to be unlabeled. Needs to be the same length as
        the number of graphs.
    node_labels : List[List[float]], optional
        The node labels of the graphs for node classification tasks. If None
        (default), the graphs are assumed to be unlabeled. Needs to be the same
        length as the number of nodes per graph.
    edge_labels : List[List[float]], optional
        The edge labels of the graphs for edge classification tasks. If None
        (default), the graphs are assumed to be unlabeled. Needs to be the same
        length as the number of edges per graph.
    dataset_name : str, optional
        The name of the dataset. If None (default), the name of the dataset is
        used.
    graph_count : int, optional
        The number of graphs to subsample. If None (default), all graphs are
        used. If the number of graphs in the dataset is smaller than
        `graph_count`, all graphs are used.
    seed : int, optional
        The random seed to use for subsampling. Default is 0.

    Returns
    -------
    dataset : Dataset
    """
    if isinstance(name_or_graphs, str):
        dataset = _load_dataset(
            name_or_graphs, labels, node_labels, edge_labels, graph_count, seed
        )
    elif isinstance(name_or_graphs, list):
        dataset = _load_graphs(name_or_graphs, labels, node_labels, edge_labels)
        if graph_count is not None and graph_count < len(dataset):
            dataset = dataset.subsample(graph_count, seed)
    elif any(
        base.__name__ == "DGLBuiltinDataset" or base.__name__ == "DGLDataset"
        for base in type(name_or_graphs).__bases__
    ):
        dataset = Dataset.from_dgl(
            name_or_graphs, labels, node_labels, edge_labels, graph_count, seed
        )
    elif any(
        base.__name__ == "InMemoryDataset" or base.__name__ == "Dataset"
        for base in type(name_or_graphs).__bases__
    ):
        dataset = Dataset.from_pyg(
            name_or_graphs, labels, node_labels, edge_labels, graph_count, seed
        )
    else:
        raise TypeError(f"Unsupported type {type(name_or_graphs)}")

    if dataset_name is not None:
        dataset.name = dataset_name
    else:
        dataset.name = name_or_graphs if isinstance(name_or_graphs, str) else None

    return dataset


def _load_dataset(
    name: str,
    labels: List = None,
    node_labels: List = None,
    edge_labels: List = None,
    graph_count: int = None,
    seed: int = 0,
) -> Dataset:
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
    edge_labels : List[List[any]], optional
        The edge labels of the graphs for link classification tasks. If None
        (default), the edge labels in the dataset are used, if available. If the
        dataset is already labeled, the edge labels override the edge labels.
    graph_count : int, optional
        The number of graphs to subsample. If None (default), all graphs are
        used. If the number of graphs in the dataset is smaller than
        `graph_count`, all graphs are used.
    seed : int, optional
        The random seed to use for subsampling. Default is 0.

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
        return Dataset.from_dgl(
            ogb_dataset, labels, node_labels, edge_labels, graph_count, seed
        )

    if name.startswith("ogbn"):
        ogb = _import_ogbn()
        ogb_dataset = DATASETS[name](ogb)
        return Dataset.from_dgl(
            ogb_dataset, labels, node_labels, edge_labels, graph_count, seed
        )

    if name.startswith("ogbl"):
        ogb = _import_ogbl()
        ogb_dataset = DATASETS[name](ogb)
        return Dataset.from_dgl(
            ogb_dataset, labels, node_labels, edge_labels, graph_count, seed
        )

    # Otherwise either a DGL or PyG dataset
    try:
        dgl = _import_dgl()
        dgl_dataset = DATASETS[name](dgl)
        return Dataset.from_dgl(
            dgl_dataset, labels, node_labels, edge_labels, graph_count, seed
        )

    except AttributeError:
        pyg = _import_pyg()
        pyg_dataset = DATASETS[name](pyg)
        return Dataset.from_pyg(
            pyg_dataset, labels, node_labels, edge_labels, graph_count, seed
        )


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


def _import_pyg():
    """Import torch_geometric."""
    try:
        import torch_geometric
    except ModuleNotFoundError:
        raise ModuleNotFoundError("PyTorch Geometric is required for PyG datasets.")
    return torch_geometric


def _import_ogbg():
    """Import ogb.graphpropped."""
    try:
        import ogb.graphproppred as ogbg
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "ogb is used for OGB datasets, which requires PyTorch to be installed."
        )
    return ogbg


def _import_ogbn():
    """Import ogb.nodepropped."""
    try:
        import ogb.nodeproppred as ogbn
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "ogb is used for OGB datasets, which requires PyTorch to be installed."
        )
    return ogbn


def _import_ogbl():
    """Import ogb.linkpropped."""
    try:
        import ogb.linkproppred as ogbl
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "ogb is used for OGB datasets, which requires PyTorch to be installed."
        )
    return ogbl


def _load_graphs(
    graphs: List[Union[nx.Graph, ig.Graph]],
    labels: List = None,
    node_labels: List = None,
    edge_labels: List = None,
) -> Dataset:
    """Load graphs.

    Parameters
    ----------
    graphs : List[Union[nx.Graph, ig.Graph]]
        The graphs.
    labels : List[any], optional
        The labels of the graphs for classification tasks. If None (default),
        the graphs are assumed to be unlabeled.
    node_labels : List[List[any]], optional
        The node labels of the graphs for node classification tasks. If None
        (default), the graphs are assumed to be unlabeled.
    edge_labels : List[List[any]], optional
        The edge labels of the graphs for link classification tasks. If None
        (default), the graphs are assumed to be unlabeled.

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

    return Dataset(graphs, labels, node_labels, edge_labels)

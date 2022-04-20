"""Get graphs from certain graph classes.

We either generate the graphs ourselves, or get it from 
http://users.cecs.anu.edu.au/~bdm/data/graphs.html. It is
possible to query for other graph classes in this repository,
but the following candidates do not include graphs with same
number of nodes, so are not included here:

- "cfi", # Cai, Fürer and Immerman Graphs
- "mz", # Miyazaki Graphs
- "cmz", # Miyazaki Graphs Variant (Bliss)
- "mz-aug", # Miyazaki Graphs Augmented (Bliss)
- "mz-aug2", # Miyazaki Graphs Augmented 2 (Bliss)
- "ag", # Affine geometry graphs
- "sts", # Steiner Triple Systems graphs
- "paley", # Paley graphs
"""
import io
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import igraph as ig
import networkx as nx
import requests

_DATA_DIR = Path(__file__).parents[1] / "data"

GRAPH_CLASSES = {
    "all": [3, 4, 5, 6, 7],  # All nonisomorphic graphs
    "pp": [4, 5, 6, 7, 8],  # Projective Plane Graphs
    "eul": [5, 6, 7],  # Eulerian graphs
    "planar_conn": [3, 4, 5, 6, 7],  # Planar Connected Graphs
    "chordal": [4, 5, 6, 7, 8],  # Chordal graphs
    "perfect": [5, 6, 7],  # Perfect graphs
    "highly_irregular": [8, 9, 10, 11, 12, 13],  # Highly irregular graphs
    "crit4": [7, 8, 9, 10],  # Edge-4-critical graphs
    "sr251256": [25],  # Strongly Regular Graphs SR(25,12,5,6)
    "sr261034": [26],  # Strongly Regular Graphs SR(26,10,3,4)
    "sr281264": [28],  # Strongly Regular Graphs SR(28,12,6,4)
    "sr291467": [29],  # Strongly Regular Graphs SR(29,14,6,7)
}

_ISOCLASS_SIZES = {
    3: 4,
    4: 11,
    5: 34,
    6: 156,
    7: 1044,
}


def get_graphs(graph_class) -> Dict[int, List[ig.Graph]]:
    """Get graphs of the given graph class, by their vertex count.

    If the graphs do not exist in the data folder,
    either download them, or generate and save them.

    Parameters
    ----------
    graph_class : str
        The graph class to get.

    Returns
    -------
    dict
        A dictionary of graph lists, indexed by their vertex count.
    """
    if graph_class not in GRAPH_CLASSES:
        raise ValueError(f"Unknown graph class: {graph_class}")

    data_subdir = _DATA_DIR / graph_class
    dir_exists = _check_data_dir(data_subdir)

    if dir_exists:
        node_counts = [int(f.stem) for f in data_subdir.iterdir() if f.is_dir()]
        graphs = {}
        for node_count in node_counts:
            data_subsubdir = data_subdir / str(node_count)
            graphs[node_count] = [
                ig.Graph.Read(str(f)) for f in data_subsubdir.iterdir()
            ]
        return graphs

    node_counts = GRAPH_CLASSES[graph_class]

    if graph_class == "all":
        graphs = {}
        for node_count in node_counts:
            graphs[node_count] = _generate_nonisomorphic_graphs(node_count)
        _save_graphs(graph_class, graphs)
        return graphs

    elif graph_class in ["chordal", "perfect", "eul", "highly_irregular"]:
        graphs = _download_graph6_graphs(graph_class, node_counts)
        _save_graphs(graph_class, graphs)
        return graphs

    elif graph_class in ["planar_conn", "crit4"]:
        graphs = _download_graph6_graphs(graph_class + ".", node_counts)
        _save_graphs(graph_class, graphs)
        return graphs

    elif graph_class.startswith("sr"):
        graphs = _download_graph6_graphs(
            graph_class, node_counts, use_node_count_in_url=False
        )
        _save_graphs(graph_class, graphs)
        return graphs

    else:
        graphs = _download_zipped_graphs(graph_class)
        _save_graphs(graph_class, graphs)
        return graphs


def _check_data_dir(dir: Path) -> bool:
    """Check if the data directory exists.

    If not, create it and return False.

    Parameters
    ----------
    graph_class : str
        The graph class to check.

    Returns
    -------
    bool
        True if the data directory exists, False otherwise.
    """
    return dir.exists() and dir.is_dir() and len(list(dir.iterdir())) != 0


def _save_graphs(graph_class: str, graphs: Dict[int, List[ig.Graph]]):
    """Save the given graphs to the data directory.

    Parameters
    ----------
    graph_class : str
        The graph class to save.
    graphs : dict
        A dictionary of graphs, indexed by their vertex count.
    """
    data_subdir = _DATA_DIR / graph_class
    data_subdir.mkdir(parents=True, exist_ok=True)

    for node_count, graphs in graphs.items():
        data_subsubdir = data_subdir / str(node_count)
        data_subsubdir.mkdir(parents=True, exist_ok=True)
        for idx, graph in enumerate(graphs):
            graph.write_pickle(str(data_subsubdir / f"{idx}.pickle"))


def _generate_nonisomorphic_graphs(
    node_count: int,
) -> List[ig.Graph]:
    """
    Generate all nonisomorphic graphs given node count.

    Parameters
    ----------
    node_count : int
        Node count of the generated graphs, between 3 and 7.

    Returns
    -------
    list
        A list of graphs.
    """
    if node_count < 3 or node_count > 7:
        raise ValueError(f"Node count must be between 3 and 7, got {node_count}")

    class_size = _ISOCLASS_SIZES[node_count]

    if node_count == 7:
        graphs = _download_graph6_graphs("graph", [7])[7]

    else:
        # Just use the precomputed Isomorphism classes
        graphs = [
            ig.Graph.Isoclass(node_count, isoclass) for isoclass in range(class_size)
        ]

    return graphs


def _download_graph6_graphs(
    graph_class: str, node_counts=[3, 4, 5, 6, 7], use_node_count_in_url=True
) -> Dict[int, List[ig.Graph]]:
    """Download the given graph class from http://users.cecs.anu.edu.au/~bdm/data/graphs.html.

    Only return the graphs that are sufficiently small (i.e. less than 1000 nodes),
    and with more than one graph per given node count.

    Parameters
    ----------
    graph_class : str
        The graph class to download.
    node_counts : list
        A list of node counts to download.

    Returns
    -------
    dict
        A dictionary of graph lists, indexed by their vertex count.
    """
    graphs = {}
    with requests.Session() as s:
        for node_count in node_counts:
            node_count_in_url = node_count if not use_node_count_in_url else ""
            link = f"http://users.cecs.anu.edu.au/~bdm/data/{graph_class}{node_count_in_url}.g6"

            with tempfile.TemporaryFile() as temp_file:
                r = s.get(link)
                temp_file.write(r.content)
                temp_file.seek(0)
                g6_graphs = nx.read_graph6(temp_file)

            graphs[node_count] = [
                ng
                for g in g6_graphs
                if (ng := ig.Graph.from_networkx(g)).vcount() < 1000
            ]

    graphs = {
        node_count: graph_list
        for node_count, graph_list in graphs.items()
        if len(graph_list) > 1
    }

    return graphs


def _download_zipped_graphs(graph_class: str) -> Dict[int, List[ig.Graph]]:
    """Download the given graph class from https://pallini.di.uniroma1.it/Graphs.html.

    Only return the graphs that are sufficiently small (i.e. less than 1000 nodes),
    and with more than one graph per given node count.

    Parameters
    ----------
    graph_class : str
        The graph class to download.

    Returns
    -------
    dict
        A dictionary of graph lists, indexed by their vertex count.
    """
    link = f"https://pallini.di.uniroma1.it/library/undirected_dim/{graph_class}.zip"
    r = requests.get(link)

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        graphs = defaultdict(list)
        for f in z.infolist():
            if zipfile.Path(z, f.filename).is_file() and "__MACOSX" not in f.filename:
                with z.open(f) as z_file:
                    graph = _parse_dimacs(z_file.readlines())
                    if graph.vcount() < 1000:
                        graphs[graph.vcount()].append(graph)

    graphs = {
        node_count: graph_list
        for node_count, graph_list in graphs.items()
        if len(graph_list) > 1
    }

    return graphs


def _parse_dimacs(dimacs_file: List[bytes]) -> ig.Graph:
    """Parse a dimacs file into an igraph graph.

    Read_DIMACS in igraph does not work for bliss file format.

    Parameters
    ----------
    dimacs_file : List[bytes]
        The lines of the DIMACS file.

    Returns
    -------
    igraph.Graph
        The parsed graph.
    """
    PROBLEM_LINE_REGEX = re.compile(r"^p\s+edge\s+(?P<vcount>\d+)\s+(?P<ecount>\d+)")
    EDGE_LINE_REGEX = re.compile(r"^e\s+(?P<source>\d+)\s+(?P<target>\d+)")

    graph = None
    edges = []
    for line in dimacs_file:
        line = line.decode("utf-8")
        if line.startswith("p"):
            problem_line = line
            match = PROBLEM_LINE_REGEX.match(problem_line)
            graph = ig.Graph(
                n=int(match.group("vcount")),
                directed=False,
            )
        elif line.startswith("e"):
            edge_info = EDGE_LINE_REGEX.match(line)
            edges.append(
                (
                    int(edge_info.group("source")) - 1,
                    int(edge_info.group("target")) - 1,
                )
            )

    graph.add_edges(edges)

    return graph

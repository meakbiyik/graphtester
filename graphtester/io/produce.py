"""Get graphs from certain graph classes.

We either generate the graphs ourselves, or get it from
https://pallini.di.uniroma1.it/Graphs.html (or alternatively
http://users.cecs.anu.edu.au/~bdm/data/graphs.html, or
https://www.distanceregular.org/indexes/upto50vertices.html). It is
possible to query for other graph classes in this repository,
but the following candidates do not have multiple graphs with same
number of nodes, so are not included here:

- "cfi", # Cai, Fürer and Immerman Graphs
- "mz", # Miyazaki Graphs
- "cmz", # Miyazaki Graphs Variant (Bliss)
- "mz-aug", # Miyazaki Graphs Augmented (Bliss)
- "mz-aug2", # Miyazaki Graphs Augmented 2 (Bliss)
- "ag", # Affine geometry graphs
- "sts", # Steiner Triple Systems graphs
- "paley", # Paley graphs

The following graph classes contain graphs that are too large to
feasibly label:

- "pp", # Projective Plane Graphs

Similarly, we omit SR(35,18,9,9) as the graph class is too
large (3854) considering the vertex count.
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
import pandas as pd
import requests
from bs4 import BeautifulSoup

_DATA_DIR = Path(__file__).parents[1] / "data"

GRAPH_CLASSES = {
    "all": [3, 4, 5, 6, 7, 8],
    "eul": [3, 4, 5, 6, 7, 8, 9],
    "planar_conn": [3, 4, 5, 6, 7, 8],
    "chordal": [4, 5, 6, 7, 8, 9],
    "perfect": [5, 6, 7, 8],
    "highlyirregular": [8, 9, 10, 11, 12, 13],
    "crit4": [7, 8, 9, 10, 11],
    "selfcomp": [5, 8, 9, 12, 13],
    "strongly_regular": [16, 25, 26, 28, 29, 36, 40, 45, 50, 64],
    # fmt: off
    "distance_regular": [
        10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25,
        26, 27, 28, 30, 32, 35, 36, 40, 42, 45, 50,
    ],
    # fmt: on
}

FAST_GRAPH_CLASSES = {
    "all": [3, 4, 5, 6, 7],
    "eul": [3, 4, 5, 6, 7, 8, 9],
    "planar_conn": [3, 4, 5, 6, 7],
    "chordal": [4, 5, 6, 7, 8],
    "perfect": [5, 6, 7],
    "highlyirregular": [8, 9, 10, 11, 12, 13],
    "crit4": [7, 8, 9, 10, 11],
    "selfcomp": [5, 8, 9, 12],
}

GRAPH_CLASS_DESCRIPTIONS = {
    "all": "All nonisomorphic",
    "eul": "Eulerian",
    "planar_conn": "Planar connected",
    "chordal": "Chordal",
    "perfect": "Perfect",
    "highlyirregular": "Highly irregular",
    "crit4": "Edge-4-critical",
    "selfcomp": "Self-complementary",
    "strongly_regular": "Strongly regular",
    "distance_regular": "Distance regular (non-exhaustive)",
}

_ISOCLASS_SIZES = {
    3: 4,
    4: 11,
    5: 34,
    6: 156,
    7: 1044,
    8: 12346,
}

_SR_LINKS = {
    16: "http://www.maths.gla.ac.uk/~es/SRGs/16-6-2-2",
    25: "http://www.maths.gla.ac.uk/~es/SRGs/25-12-5-6",
    26: "http://www.maths.gla.ac.uk/~es/SRGs/26-10-3-4",
    28: "http://www.maths.gla.ac.uk/~es/SRGs/28-12-6-4",
    29: "http://www.maths.gla.ac.uk/~es/SRGs/29-14-6-7",
    35: "http://www.maths.gla.ac.uk/~es/SRGs/35-18-9-9",
    36: "http://www.maths.gla.ac.uk/~es/SRGs/36-14-4-6",
    40: "http://www.maths.gla.ac.uk/~es/SRGs/40-12-2-4",
    45: "http://www.maths.gla.ac.uk/~es/SRGs/45-12-3-3",
    50: "http://www.maths.gla.ac.uk/~es/SRGs/50-21-8-9",
    64: "http://www.maths.gla.ac.uk/~es/SRGs/64-18-2-6",
}


def produce(graph_class, max_node_count=None) -> Dict[int, List[ig.Graph]]:
    """Produce graphs of the given graph class, by their vertex count.

    If the graphs do not exist in the data folder,
    either download them, or generate and save them.

    Following graph classes are supported (for the given node counts):
        - "all": All non-isomorphic graphs for given node sizes - [3, 4, 5, 6, 7, 8]
        - "eul": Eulerian graphs - [3, 4, 5, 6, 7, 8, 9]
        - "planar_conn": Planar connected graphs - [3, 4, 5, 6, 7, 8]
        - "chordal": Chordal graphs - [4, 5, 6, 7, 8, 9]
        - "perfect": Perfect graphs - [5, 6, 7, 8]
        - "highlyirregular": Highly irregular graphs - [8, 9, 10, 11, 12, 13]
        - "crit4": Edge-4-critical graphs - [7, 8, 9, 10, 11]
        - "selfcomp": Self-complementary graphs - [5, 8, 9, 12, 13]
        - "strongly_regular": Strongly regular graphs -
            [16, 25, 26, 28, 29, 36, 40, 45, 50, 64]
        - "distance_regular": Distance regular graphs -
            [10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25,
             26, 27, 28, 30, 32, 35, 36, 40, 42, 45, 50]

    The graphs in their respective classes and node counts are exhaustive,
    with the exception of "distance_regular" graphs.

    Parameters
    ----------
    graph_class : str
        The graph class to get.
    max_node_count : int
        The maximum number of nodes per graph. If None, all graphs are
        returned.

    Returns
    -------
    dict
        A dictionary of graph lists, indexed by their vertex count.
    """
    if graph_class not in GRAPH_CLASSES:
        raise ValueError(f"Unknown graph class: {graph_class}")

    if max_node_count is None:
        max_node_count = 1e6

    data_subdir = _DATA_DIR / graph_class
    dir_exists = _check_data_dir(data_subdir, graph_class)

    if dir_exists:
        return _get_graphs_from_cache(data_subdir, max_node_count)

    node_counts = GRAPH_CLASSES[graph_class]

    if graph_class == "all":
        graphs = {}
        for node_count in node_counts:
            graphs[node_count] = _generate_nonisomorphic_graphs(node_count)

    elif graph_class in ["chordal", "perfect", "eul", "highlyirregular", "selfcomp"]:
        graphs = _download_graph6_graphs(graph_class, node_counts)

    elif graph_class in ["planar_conn", "crit4"]:
        graphs = _download_graph6_graphs(graph_class + ".", node_counts)

    elif graph_class == "strongly_regular":
        graphs = _download_sr_graphs(node_counts)

    elif graph_class == "distance_regular":
        graphs = _download_distance_regular_graphs(node_counts)

    else:
        graphs = _download_zipped_graphs(graph_class)

    _save_graphs(graph_class, graphs)

    return {
        node_count: graph_list
        for node_count, graph_list in graphs.items()
        if node_count <= max_node_count
    }


def _check_data_dir(dir: Path, graph_class: str) -> bool:
    """Check if the data directory exists and is sane.

    If not, create it and return False.

    Parameters
    ----------
    dir : Path
        The directory to check.
    graph_class : str
        The graph class to check.

    Returns
    -------
    bool
        True if the data directory exists, False otherwise.
    """
    if not (dir.exists() and dir.is_dir()):
        return False

    node_counts = GRAPH_CLASSES[graph_class]
    saved_node_counts = sorted(
        [int(f.stem) for f in dir.iterdir() if f.is_dir() and f.stem.isdigit()]
    )
    return saved_node_counts == node_counts


def _get_graphs_from_cache(
    data_subdir: Path, max_node_count: int
) -> Dict[int, List[ig.Graph]]:
    """Get graphs from the cache.

    Parameters
    ----------
    data_subdir : Path
        The directory to get the graphs from.
    max_node_count : int
        The maximum number of nodes per graph.

    Returns
    -------
    dict
        A dictionary of graph lists, indexed by their vertex count.
    """
    node_counts = sorted([int(f.stem) for f in data_subdir.iterdir() if f.is_dir()])
    graphs = {}
    for node_count in node_counts:
        if node_count <= max_node_count:
            data_subsubdir = data_subdir / str(node_count)
            graphs[node_count] = [
                ig.Graph.Read(str(f)) for f in data_subsubdir.iterdir()
            ]

    return graphs


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
    if node_count < 3 or node_count > 8:
        raise ValueError(f"Node count must be between 3 and 8, got {node_count}")

    class_size = _ISOCLASS_SIZES[node_count]

    if node_count >= 7:
        graphs = _download_graph6_graphs("graph", [node_count])[node_count]

    else:
        # Just use the precomputed Isomorphism classes
        graphs = [
            ig.Graph.Isoclass(node_count, isoclass) for isoclass in range(class_size)
        ]

    return graphs


def _download_sr_graphs(node_counts: List[int]) -> Dict[int, List[ig.Graph]]:
    """
    Download all strongly regular graphs of given node counts.

    Parameters
    ----------
    node_counts : List[int]
        List of node counts.

    Returns
    -------
    dict
        A dictionary of graphs, indexed by their vertex count.
    """
    graphs = {}
    for node_count in node_counts:

        link = _SR_LINKS[node_count]

        page_content = _download_content(link)
        graph_list = _parse_incidence_matrices(page_content.decode("utf-8"))
        graphs[node_count] = graph_list

    return graphs


def _download_graph6_graphs(
    graph_class: str,
    node_counts: List[int] = [3, 4, 5, 6, 7],
) -> Dict[int, List[ig.Graph]]:
    """Download the given graph class from Brendan McKay's repository.

    See http://users.cecs.anu.edu.au/~bdm/data/graphs.html.

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
    for node_count in node_counts:
        link = (
            f"http://users.cecs.anu.edu.au/~bdm/data/" f"{graph_class}{node_count}.g6"
        )

        with tempfile.TemporaryFile() as temp_file:
            content = _download_content(link)
            temp_file.write(content)
            temp_file.seek(0)
            g6_graphs = nx.read_graph6(temp_file)

        graphs[node_count] = [
            ng for g in g6_graphs if (ng := ig.Graph.from_networkx(g)).vcount() < 1000
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
    content = _download_content(link)

    with zipfile.ZipFile(io.BytesIO(content)) as z:
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


def _download_distance_regular_graphs(
    node_counts: List[int],
) -> Dict[int, List[ig.Graph]]:
    """Download the given graph class from https://www.distanceregular.org/.

    Parameters
    ----------
    node_counts : List[int]
        The graph class to download.

    Returns
    -------
    dict
        A dictionary of graph lists, indexed by their vertex count.
    """
    graph_table = _get_dist_regular_graph_table()
    graphs = {}
    for node_count in node_counts:
        links = graph_table[graph_table["No. of vertices"] == node_count]["Link"]
        graphs[node_count] = sum([_parse_graphs_from_link(link) for link in links], [])

    return graphs


def _get_dist_regular_graph_table():
    """Parse the distance regular graph table in https://www.distanceregular.org/.

    Links in the table then will be used to download the graphs.

    Returns
    -------
    pandas.DataFrame
        The distance regular graph table.
    """
    url = "https://www.distanceregular.org/indexes/upto50vertices.html"
    df = pd.read_html(
        url,
        header=0,
        converters={
            "No. of vertices": lambda x: int(x.replace("$", "")),
        },
    )[0]

    page_content = _download_content(url)
    soup = BeautifulSoup(page_content.decode("utf-8"), "lxml")
    table = soup.find("table")
    table_rows = table.findAll("tr")[1:]  # Omit the header

    links = []
    for row in table_rows:
        link_cell = row.find("td")
        link = link_cell.find("a")["href"].replace(
            "..", "https://www.distanceregular.org"
        )
        links.append(link)

    df["Link"] = links

    return df


def _parse_graphs_from_link(link: str) -> List[ig.Graph]:
    """Find and read the graphs in the given page.

    Parses the content of the link, finds links to adjacency matrix
    files and reads them into igraph Graphs.

    Parameters
    ----------
    link : str
        The link to parse the graphs from.

    Returns
    -------
    list
        The parsed graphs.
    """
    if link == "https://www.distanceregular.org/graphs/complement-srg26.10.3.4.html":
        # At the time of this analysis, above link was dead. We skip it explicitly.
        return []

    page_content = _download_content(link)
    soup = BeautifulSoup(page_content.decode("utf-8"), "lxml")
    data_links = [
        a["href"].replace("..", "https://www.distanceregular.org")
        for a in soup.find_all("a", text=re.compile(r"Adjacency matri(x|ces)}"))
    ]

    graphs = []
    for data_link in data_links:
        if data_link in [
            "https://www.distanceregular.org/graphdata/ig-15-7-3.am",
            "https://www.distanceregular.org/graphdata/ig-15-8-4.am",
        ]:
            # At the time of this analysis, above links were dead.
            # We skip them explicitly.
            continue
        page_content = _download_content(data_link)
        graph_list = _parse_incidence_matrices(page_content.decode("utf-8"))
        graphs.extend(graph_list)

    return graphs


def _parse_incidence_matrices(incidence_matrices: str) -> List[ig.Graph]:
    """Parse a string of incidence matrices into a list of graphs.

    Parameters
    ----------
    incidence_matrices : str
        The string of incidence matrices to parse.

    Returns
    -------
    list
        A list of graphs.
    """
    splitter = re.compile(r"(\r?\n){2,}")
    graph_strings = splitter.split(incidence_matrices)
    adj_matrices = [
        graph_string.strip().splitlines()
        for graph_string in graph_strings
        if graph_string.strip()
    ]

    graphs = []
    for mat_str in adj_matrices:
        if len(mat_str) < 2:
            continue
        mat_bool = [
            [int(elem) for elem in list(row.strip())]
            for row in mat_str
            if row.strip().startswith(("0", "1"))
        ]
        graphs.append(ig.Graph.Adjacency(mat_bool, mode="undirected"))

    return graphs


def _download_content(link: str) -> bytes:
    """Download the given link.

    Handle HTTP errors.

    Parameters
    ----------
    link : str
        The link to download.

    Returns
    -------
    bytes
        The downloaded content.
    """
    with requests.Session() as s:
        r = s.get(link)

    if r.status_code != 200:
        raise requests.HTTPError(
            f"Could not download graphs from the address "
            f"{link}. HTTP error {r.status_code}"
        )

    return r.content


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

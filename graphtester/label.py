"""Deterministically label and rewire graphs to make them 1-WL-distinguishable."""
from typing import List

import igraph as ig


def label_graph(graph: ig.Graph, methods: List[str], copy: bool = True) -> ig.Graph:
    """Deterministically label and rewire a graph.

    Uses the given provided methods. Compress the
    labels as string and add as a "label" attribute.

    Parameters
    ----------
    graph : ig.Graph
        The graph to label and rewire.
    methods : List[str]
        The methods to use to label the graph.
    copy : bool
        Whether to operate on a copy of the graph.

    Returns
    -------
    ig.Graph
        The labeled and rewired graph.
    """
    edge_rewirers, edge_labelers, vertex_labelers = [], [], []
    for method in methods:
        if method in VERTEX_LABELING_METHODS:
            vertex_labelers.append(VERTEX_LABELING_METHODS[method])
        elif method in EDGE_LABELING_METHODS:
            edge_labelers.append(EDGE_LABELING_METHODS[method])
        elif method in EDGE_REWIRING_METHODS:
            edge_rewirers.append(EDGE_REWIRING_METHODS[method])
        else:
            raise ValueError(f"Unknown labeling method: {method}")

    if copy:
        graph = graph.copy()

    # Apply vertex labeling
    vertex_label_lists = [method(graph) for method in vertex_labelers]
    compressed_vertex_labels = (
        ["".join(labels) for labels in zip(*vertex_label_lists)]
        if vertex_labelers
        else [""] * graph.vcount()
    )

    # Apply edge labeling
    edge_label_lists = [method(graph) for method in edge_labelers]
    compressed_edge_labels = (
        ["".join(labels) for labels in zip(*edge_label_lists)]
        if edge_labelers
        else [""] * graph.ecount()
    )

    graph.vs["label"] = compressed_vertex_labels
    graph.es["label"] = compressed_edge_labels

    # Apply edge rewiring
    for method in edge_rewirers:
        method(graph)

    return graph


def _neighborhood_subgraph_component_count(graph: ig.Graph) -> List[str]:
    """Compute the number of components in each neighborhood subgraph.

    Parameters
    ----------
    graph : ig.Graph
        The graph to label.

    Returns
    -------
    List[str]
        The labels.
    """
    return [
        str(
            len(
                graph.induced_subgraph(
                    graph.neighborhood(node_idx, mindist=1)
                ).decompose(mode="weak")
            )
        )
        for node_idx in range(graph.vcount())
    ]


def _neighborhood_subgraph_component_sizes(graph: ig.Graph) -> List[str]:
    """Compute the sizes of components in each neighborhood subgraph.

    Parameters
    ----------
    graph : ig.Graph
        The graph to label.

    Returns
    -------
    List[str]
        The labels.
    """
    return [
        ";".join(
            sorted(
                [
                    str(comp.vcount())
                    for comp in graph.induced_subgraph(
                        graph.neighborhood(node_idx, mindist=1)
                    ).decompose(mode="weak")
                ]
            )
        )
        for node_idx in range(graph.vcount())
    ]


def _neighborhood_subgraph_component_signatures(graph: ig.Graph) -> List[str]:
    """Compute the WL signatures of components in each neighborhood subgraph.

    Uses edge betweenness to create a signature.

    Parameters
    ----------
    graph : ig.Graph
        The graph to label.

    Returns
    -------
    List[str]
        The labels.
    """
    return [
        ";".join(
            sorted(
                [
                    ",".join(
                        [str(round(b, 6)) for b in sorted(comp.edge_betweenness())]
                    )
                    for comp in graph.induced_subgraph(
                        graph.neighborhood(node_idx, mindist=1)
                    ).decompose(mode="weak")
                ]
            )
        )
        for node_idx in range(graph.vcount())
    ]


VERTEX_LABELING_METHODS = {
    # "coreness": lambda g: [str(round(h, 6)) for h in g.coreness()],
    "eigenvector": lambda g: [str(round(h, 6)) for h in g.evcent()],
    "eccentricity": lambda g: [str(round(h, 6)) for h in g.eccentricity()],
    "local_transitivity": lambda g: [
        str(round(h, 6)) for h in g.transitivity_local_undirected(mode="zero")
    ],
    "harmonic": lambda g: [str(round(h, 6)) for h in g.harmonic_centrality()],
    "closeness": lambda g: [str(round(h, 6)) for h in g.closeness()],
    # "pagerank": lambda g: [str(round(h, 6)) for h in g.pagerank()],
    # "hub_score": lambda g: [str(round(h, 6)) for h in g.hub_score()],
    "two_hop_neighborhood_size": lambda g: [
        str(round(h, 6)) for h in g.neighborhood_size(order=2)
    ],
    # "average_neighborhood_degree": lambda g: [str(round(h, 6)) for h in g.knn()[0]],
    "burt_constraint": lambda g: [str(round(h, 6)) for h in g.constraint()],
    "betweenness": lambda g: [str(round(h, 6)) for h in g.betweenness()],
    "nbhood_subgraph_comp_count": _neighborhood_subgraph_component_count,
    "nbhood_subgraph_comp_sizes": _neighborhood_subgraph_component_sizes,
    "nbhood_subgraph_comp_sign": _neighborhood_subgraph_component_signatures,
}

EDGE_LABELING_METHODS = {
    "convergence_degree": lambda g: [str(round(h, 6)) for h in g.convergence_degree()],
    "edge_betweenness": lambda g: [
        str(round(b, 6)) for b in g.edge_betweenness(directed=False)
    ],
}

EDGE_REWIRING_METHODS = {}

ALL_METHODS = (
    list(VERTEX_LABELING_METHODS)
    + list(EDGE_LABELING_METHODS)
    + list(EDGE_REWIRING_METHODS)
)

METHOD_DESCRIPTIONS = {
    "eigenvector": "Eigenvector centrality",
    "eccentricity": "Eccentricity",
    "local_transitivity": "Local transitivity",
    "harmonic": "Harmonic centrality",
    "closeness": "Closeness centrality",
    "two_hop_neighborhood_size": "Two-hop neighborhood size",
    "burt_constraint": "Burt's constraint",
    "betweenness": "Betweenness centrality",
    "nbhood_subgraph_comp_count": "No of neighborhood subgraph components",
    "nbhood_subgraph_comp_sizes": "Neighborhood subgraph component sizes",
    "nbhood_subgraph_comp_sign": "Neighborhood subgraph component signatures",
    "convergence_degree": "Convergence degree",
    "edge_betweenness": "Edge betweenness",
}

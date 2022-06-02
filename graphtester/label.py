"""Deterministically label and rewire graphs to make them 1-WL-distinguishable."""
from collections import Counter, defaultdict
from typing import List

import igraph as ig

from .test import weisfeiler_lehman_hash


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


def _local_graph_component_count(graph: ig.Graph) -> List[str]:
    """Compute the number of components in each first subconstituent.

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


def _local_graph_component_sizes(graph: ig.Graph) -> List[str]:
    """Compute the sizes of components in each first subconstituent.

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


def _neighborhood_1st_subconst_sign(graph: ig.Graph) -> List[str]:
    """Compute the WL signatures of each node using 1st subconstituents.

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
                        [
                            str(round(b, 6))
                            for b in sorted(
                                graph.induced_subgraph(
                                    graph.neighborhood(node_idx, mindist=1)
                                ).edge_betweenness()
                            )
                        ]
                    )
                ]
            )
        )
        for node_idx in range(graph.vcount())
    ]


def _neighborhood_2nd_subconst_sign(graph: ig.Graph) -> List[str]:
    """Compute the WL signatures of each node using 2nd subconstituents.

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
                        [
                            str(round(b, 6))
                            for b in sorted(
                                graph.induced_subgraph(
                                    graph.neighborhood(node_idx, mindist=2, order=2)
                                ).edge_betweenness()
                            )
                        ]
                    )
                ]
            )
        )
        for node_idx in range(graph.vcount())
    ]


def _count_substructure_vertices(
    graph: ig.Graph,
    substructure: ig.Graph,
    substructure_vertex_orbits: List[str] = None,
) -> List[str]:
    """Count the substructure occurence in a graph.

    Return the number of times each vertex has been in an orbit
    of the substructure in the original graph, as a string label
    per vertex.

    Implements the approach described in [1].

    Parameters
    ----------
    graph : ig.Graph
        The graph to label.
    substructure : ig.Graph
        The substructure to count.
    substructure_vertex_orbits : List[str]
        The vertex orbits of the substructure.

    Returns
    -------
    List[str]
        The labels.

    References
    ----------
    [1] Bouritsas et al., Improving Graph Neural Network Expressivity via
        Subgraph Isomorphism Counting, 2021.
    """
    if substructure_vertex_orbits is None:
        substructure_vertex_orbits = _determine_vertex_orbits(substructure)

    subisomorphisms = graph.get_subisomorphisms_lad(substructure)

    if not subisomorphisms:
        return [""] * graph.vcount()

    matched_vertices = [
        Counter(vertices_mapped_to_substructure_node)
        for vertices_mapped_to_substructure_node in zip(*subisomorphisms)
    ]

    matched_orbits = defaultdict(Counter)
    for idx, orbit in enumerate(substructure_vertex_orbits):
        matched_orbits[orbit] += matched_vertices[idx]

    return [
        ",".join([str(orbit_counter[i]) for orbit_counter in matched_orbits.values()])
        for i in range(graph.vcount())
    ]


def _determine_vertex_orbits(substructure: ig.Graph) -> List[str]:
    """Determine the orbits of substructure vertices.

    The orbit of a vertex v is the set of vertices to which it
    can be mapped via an automorphism. See [1] for more details.

    Parameters
    ----------
    substructure : ig.Graph
        The substructure to determine the orbits of.

    Returns
    -------
    List[str]
        The orbit labels.

    References
    ----------
    [1] Bouritsas et al., Improving Graph Neural Network Expressivity via
        Subgraph Isomorphism Counting, 2021.
    """
    automorphisms = substructure.get_isomorphisms_vf2()
    orbits = [tuple(sorted(set(all_mappings))) for all_mappings in zip(*automorphisms)]
    orbit_hashmap = {orbit: str(i) for i, orbit in enumerate(set(orbits))}
    return [orbit_hashmap[orbit] for orbit in orbits]


def _count_substructure_edges(
    graph: ig.Graph, substructure: ig.Graph, substructure_edge_orbits: List[str] = None
) -> List[str]:
    """Count the substructure occurence in a graph.

    Return the number of times each edge has been in an orbit
    of the substructure in the original graph, as a string label
    per edge.

    Implements the approach described in [1].

    Parameters
    ----------
    graph : ig.Graph
        The graph to label.
    substructure : ig.Graph
        The substructure to count.
    substructure_edge_orbits : List[str], optional
        The orbits of the substructure edges.

    Returns
    -------
    List[str]
        The labels.

    References
    ----------
    [1] Bouritsas et al., Improving Graph Neural Network Expressivity via
        Subgraph Isomorphism Counting, 2021.
    """
    if substructure_edge_orbits is None:
        substructure_edge_orbits = _determine_edge_orbits(substructure)

    subisomorphisms = graph.get_subisomorphisms_lad(substructure)

    if not subisomorphisms:
        return [""] * graph.ecount()

    edge_subisomorphisms = [
        [
            (subisomorphism[edge.source], subisomorphism[edge.target])
            for edge in substructure.es
        ]
        for subisomorphism in subisomorphisms
    ]
    matched_edges = [
        Counter(edges_mapped_to_substructure_edge)
        for edges_mapped_to_substructure_edge in zip(*edge_subisomorphisms)
    ]

    matched_orbits = defaultdict(Counter)
    for idx, orbit in enumerate(substructure_edge_orbits):
        matched_orbits[orbit] += matched_edges[idx]

    return [
        ",".join(
            [
                str(orbit_counter[(edge.source, edge.target)])
                for orbit_counter in matched_orbits.values()
            ]
        )
        for edge in graph.es
    ]


def _determine_edge_orbits(substructure: ig.Graph) -> List[str]:
    """Determine the orbits of substructure edges.

    The orbit of an edge e is the set of edges to which it
    can be mapped via an automorphism. See [1] for more details.

    Parameters
    ----------
    substructure : ig.Graph
        The substructure to determine the orbits of.

    Returns
    -------
    List[str]
        The orbit labels.

    References
    ----------
    [1] Bouritsas et al., Improving Graph Neural Network Expressivity via
        Subgraph Isomorphism Counting, 2021.
    """
    automorphisms = substructure.get_isomorphisms_vf2()
    edge_automorphisms = [
        [
            (automorphism[edge.source], automorphism[edge.target])
            for edge in substructure.es
        ]
        for automorphism in automorphisms
    ]
    orbits = [
        tuple(sorted(set(all_mappings))) for all_mappings in zip(*edge_automorphisms)
    ]
    orbit_hashmap = {orbit: str(i) for i, orbit in enumerate(set(orbits))}
    return [orbit_hashmap[orbit] for orbit in orbits]


def _wl_hash_vertex_label(graph: ig.Graph) -> List[str]:
    """Label vertices by the 1-WL hash of the graph, after marking each.

    Parameters
    ----------
    graph : ig.Graph
        The graph to label.

    Returns
    -------
    List[str]
        The labels.
    """
    labels = []
    for node_idx in range(graph.vcount()):
        graph.vs["__marked"] = "0"
        graph.vs[node_idx]["__marked"] = "1"
        labels.append(weisfeiler_lehman_hash(graph, None, "__marked"))

    del graph.vs["__marked"]

    return labels


def _wl_hash_edge_label(graph: ig.Graph) -> List[str]:
    """Label edges by the 1-WL hash of the graph, after marking each.

    Parameters
    ----------
    graph : ig.Graph
        The graph to label.

    Returns
    -------
    List[str]
        The labels.
    """
    labels = []
    for edge in graph.es:
        graph.es["__marked"] = "0"
        edge["__marked"] = "1"
        labels.append(weisfeiler_lehman_hash(graph, "__marked", None))

    del graph.es["__marked"]

    return labels


def _rewire_by_edge_betweenness(graph: ig.Graph):
    """Add edge candidates with highest betweenness.

    Parameters
    ----------
    graph : ig.Graph
        The graph to rewire.

    Returns
    -------
    None
    """
    complementer = graph.complementer(loops=False)
    edge_betweennesses = []
    for edge in complementer.es:
        copy_graph = graph.copy()
        copy_graph.add_edge(edge.source, edge.target)
        edge_betweennesses.append(
            round(copy_graph.edge_betweenness(directed=False)[-1], 6)
        )

    maximum_eb = max(edge_betweennesses)
    graph.add_edges(
        [
            (edge.source, edge.target)
            for edge, betw in zip(complementer.es, edge_betweennesses)
            if betw == maximum_eb
        ]
    )


SUBSTRUCTURES = {
    "3_cycle": ig.Graph.Ring(3),
    "4_cycle": ig.Graph.Ring(4),
    "5_cycle": ig.Graph.Ring(5),
    "6_cycle": ig.Graph.Ring(6),
    "3_path": ig.Graph.Ring(3, circular=False),
    "4_path": ig.Graph.Ring(4, circular=False),
    "5_path": ig.Graph.Ring(5, circular=False),
    "6_path": ig.Graph.Ring(6, circular=False),
    "3_clique": ig.Graph.Full(3),
    "4_clique": ig.Graph.Full(4),
    "5_clique": ig.Graph.Full(5),
    "6_clique": ig.Graph.Full(6),
}

SUBSTRUCTURE_VERTEX_ORBITS = {
    name: _determine_vertex_orbits(substructure)
    for name, substructure in SUBSTRUCTURES.items()
}

SUBSTRUCTURE_EDGE_ORBITS = {
    name: _determine_edge_orbits(substructure)
    for name, substructure in SUBSTRUCTURES.items()
}


VERTEX_LABELING_METHODS = {
    "Eigenvector centrality": lambda g: [str(round(h, 6)) for h in g.evcent()],
    "Eccentricity": lambda g: [str(round(h, 6)) for h in g.eccentricity()],
    "Local transitivity": lambda g: [
        str(round(h, 6)) for h in g.transitivity_local_undirected(mode="zero")
    ],
    "Harmonic centrality": lambda g: [
        str(round(h, 6)) for h in g.harmonic_centrality()
    ],
    "Closeness centrality": lambda g: [str(round(h, 6)) for h in g.closeness()],
    "Two-hop neighborhood size": lambda g: [
        str(round(h, 6)) for h in g.neighborhood_size(order=2)
    ],
    "Burt's constraint": lambda g: [str(round(h, 6)) for h in g.constraint()],
    "Betweenness centrality": lambda g: [str(round(h, 6)) for h in g.betweenness()],
    "Marked WL hash  vertex label": _wl_hash_vertex_label,
    "3-cycle count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["3_cycle"], SUBSTRUCTURE_VERTEX_ORBITS["3_cycle"]
    ),
    "4-cycle count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["4_cycle"], SUBSTRUCTURE_VERTEX_ORBITS["4_cycle"]
    ),
    "5-cycle count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["5_cycle"], SUBSTRUCTURE_VERTEX_ORBITS["5_cycle"]
    ),
    "6-cycle count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["6_cycle"], SUBSTRUCTURE_VERTEX_ORBITS["6_cycle"]
    ),
    "3-path count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["3_path"], SUBSTRUCTURE_VERTEX_ORBITS["3_path"]
    ),
    "4-path count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["4_path"], SUBSTRUCTURE_VERTEX_ORBITS["4_path"]
    ),
    "5-path count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["5_path"], SUBSTRUCTURE_VERTEX_ORBITS["5_path"]
    ),
    "6-path count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["6_path"], SUBSTRUCTURE_VERTEX_ORBITS["6_path"]
    ),
    "3-clique count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["3_clique"], SUBSTRUCTURE_VERTEX_ORBITS["3_clique"]
    ),
    "4-clique count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["4_clique"], SUBSTRUCTURE_VERTEX_ORBITS["4_clique"]
    ),
    "5-clique count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["5_clique"], SUBSTRUCTURE_VERTEX_ORBITS["5_clique"]
    ),
    "6-clique count of vertices": lambda g: _count_substructure_vertices(
        g, SUBSTRUCTURES["6_clique"], SUBSTRUCTURE_VERTEX_ORBITS["6_clique"]
    ),
    "Local graph component count": _local_graph_component_count,
    "Local graph component sizes": _local_graph_component_sizes,
    "Neighborhood 1st subconstituent signatures": _neighborhood_1st_subconst_sign,
    "Neighborhood 2nd subconstituent signatures": _neighborhood_2nd_subconst_sign,
}

EDGE_LABELING_METHODS = {
    "Convergence degree": lambda g: [str(round(h, 6)) for h in g.convergence_degree()],
    "Edge betweenness": lambda g: [
        str(round(b, 6)) for b in g.edge_betweenness(directed=False)
    ],
    "Marked WL hash edge label": _wl_hash_edge_label,
    "3-cycle count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["3_cycle"], SUBSTRUCTURE_EDGE_ORBITS["3_cycle"]
    ),
    "4-cycle count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["4_cycle"], SUBSTRUCTURE_EDGE_ORBITS["4_cycle"]
    ),
    "5-cycle count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["5_cycle"], SUBSTRUCTURE_EDGE_ORBITS["5_cycle"]
    ),
    "6-cycle count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["6_cycle"], SUBSTRUCTURE_EDGE_ORBITS["6_cycle"]
    ),
    "3-path count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["3_path"], SUBSTRUCTURE_EDGE_ORBITS["3_path"]
    ),
    "4-path count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["4_path"], SUBSTRUCTURE_EDGE_ORBITS["4_path"]
    ),
    "5-path count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["5_path"], SUBSTRUCTURE_EDGE_ORBITS["5_path"]
    ),
    "6-path count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["6_path"], SUBSTRUCTURE_EDGE_ORBITS["6_path"]
    ),
    "3-clique count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["3_clique"], SUBSTRUCTURE_EDGE_ORBITS["3_clique"]
    ),
    "4-clique count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["4_clique"], SUBSTRUCTURE_EDGE_ORBITS["4_clique"]
    ),
    "5-clique count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["5_clique"], SUBSTRUCTURE_EDGE_ORBITS["5_clique"]
    ),
    "6-clique count of edges": lambda g: _count_substructure_edges(
        g, SUBSTRUCTURES["6_clique"], SUBSTRUCTURE_EDGE_ORBITS["6_clique"]
    ),
}

EDGE_REWIRING_METHODS = {
    "Rewire by edge betweenness": _rewire_by_edge_betweenness,
}

ALL_METHODS = (
    list(VERTEX_LABELING_METHODS)
    + list(EDGE_LABELING_METHODS)
    + list(EDGE_REWIRING_METHODS)
)

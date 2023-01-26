"""Use WL to test the graph isomorphism."""
import itertools
from typing import List

import igraph as ig


def weisfeiler_lehman_test(
    G1: ig.Graph,
    G2: ig.Graph,
    edge_attr: str | List[str] = None,
    node_attr: str | List[str] = None,
    iterations: int = None,
) -> bool:
    """Apply 1-Weisfeiler Lehman (1-WL) graph isomorphism test.

    Adapted from networkx.weisfeiler_lehman_graph_hash

    Parameters
    ----------
    G1 : ig.Graph
        The first graph.
    G2 : ig.Graph
        The second graph.
    edge_attr : str | List[str], optional
        The edge attribute(s) to use for the edge labels.
    node_attr : str | List[str], optional
        The node attribute(s) to use for the node labels.
    iterations : int, optional
        The number of iterations to run. If None (default), run (n-1) iterations,
        where n is the number of vertices of inputs. See [1] for more details.

    Returns
    -------
    bool
        True if the graphs are isomorphic, False otherwise.

    References
    ----------
    [1] Sandra Kiefer, Power and Limits of the Weisfeiler-Leman Algorithm, 2020.
    """
    if iterations is None:
        iterations = G1.vcount() - 1

    node_labels_g1 = _init_node_labels(G1, node_attr)
    node_labels_g2 = _init_node_labels(G2, node_attr)

    for _ in range(iterations):

        new_labels_g1 = _weisfeiler_lehman_step(G1, node_labels_g1, edge_attr=edge_attr)
        new_labels_g2 = _weisfeiler_lehman_step(G2, node_labels_g2, edge_attr=edge_attr)

        prev_labels_g1 = node_labels_g1
        node_labels_g1, node_labels_g2 = _reassign_labels(new_labels_g1, new_labels_g2)

        if sorted(node_labels_g1) != sorted(node_labels_g2):
            return False

        if prev_labels_g1 == node_labels_g1:
            break

    return True


def k_weisfeiler_lehman_test(
    G1: ig.Graph,
    G2: ig.Graph,
    k: int,
    edge_attr: str | List[str] = None,
    node_attr: str | List[str] = None,
    iterations: int = None,
    folklore: bool = False,
):
    """Apply k-Weisfeiler Lehman (k-WL) graph isomorphism test.

    We follow the algorithms in [2]. Note that k-WL with k=1 is not the
    same as 1-WL. In fact, 1-WL=2-WL for non-folklore implementation [2].

    Folklore WL (k-FWL) can be used instead of k-WL by using the flag "folklore".
    k-WL is equivalent in power with (k-1)-FWL, and k-FWL is also more efficient.

    Parameters
    ----------
    G1 : ig.Graph
        The first graph.
    G2 : ig.Graph
        The second graph.
    k : int
        Degree of the WL test, 2 <= k <= 6.
    edge_attr : str | List[str], optional
        The edge attribute(s) to use for the edge labels.
    node_attr : str | List[str], optional
        The node attribute(s) to use for the node labels.
    iterations : int, optional
        The number of iterations to run. If None (default), run (n**k-1) iterations,
        where n is the number of vertices of inputs. See [1] for more details.
    folklore : bool, optional
        Whether to use the folklore variant (k-FWL) of k-WL test. By default False.
        See [2] for more details.

    Returns
    -------
    bool
        True if the graphs are isomorphic, False otherwise.

    References
    ----------
    [1] Sandra Kiefer, Power and Limits of the Weisfeiler-Leman Algorithm, 2020.
    [2] Huang et al., A Short Tutorial on The Weisfeiler-Lehman Test And Its Variants,
        2021. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9413523
    """
    if k < 2 or k > 6:
        raise ValueError(f"k must be an integer between 2 and 6 inclusive, not {k}")

    if iterations is None:
        iterations = G1.vcount() ** k - 1

    node_labels_g1 = _init_k_tuple_labels(G1, k, edge_attr, node_attr)
    node_labels_g2 = _init_k_tuple_labels(G2, k, edge_attr, node_attr)

    for _ in range(iterations):

        new_labels_g1 = _k_weisfeiler_lehman_step(G1, node_labels_g1, k, folklore)
        new_labels_g2 = _k_weisfeiler_lehman_step(G2, node_labels_g2, k, folklore)

        prev_labels_g1 = node_labels_g1
        node_labels_g1, node_labels_g2 = _reassign_labels(new_labels_g1, new_labels_g2)

        if sorted(node_labels_g1) != sorted(node_labels_g2):
            return False

        if prev_labels_g1 == node_labels_g1:
            break

    return True


def weisfeiler_lehman_hash(
    G: ig.Graph,
    edge_attr: str | List[str] = None,
    node_attr: str | List[str] = None,
    iterations: int = None,
    return_graph: bool = False,
    return_hash: bool = True,
):
    """Apply 1-Weisfeiler Lehman (1-WL) test to create a graph hash.

    Parameters
    ----------
    G : ig.Graph
        The graph.
    edge_attr : str | List[str], optional
        The edge attribute(s) to use for the edge labels.
    node_attr : str | List[str], optional
        The node attribute(s) to use for the node labels.
    iterations : int, optional
        The maximum number of iterations to run. If None (default), run (n-1)
        iterations where n is the number of vertices of inputs. See [1] for
        more details.
    return_graph : bool, optional
        Whether to return a copy of the graph with the updated node labels.
        By default False. This is useful to check the node labels, and run
        more iterations on the same graph if needed.
    return_hash : bool, optional
        Whether to return the hash of the graph. By default True. If False,
        the first return value is None. This is useful if the graph is
        too large to concatenate the node labels into a string.

    Returns
    -------
    str
        The hash of the graph.
    ig.Graph, optional
        The graph with updated node labels. Only returned if `return_graph` is True.
        The label is stored in the node attribute "label", other node attributes are
        removed.

    References
    ----------
    [1] Sandra Kiefer, Power and Limits of the Weisfeiler-Leman Algorithm, 2020.
    """
    if iterations is None:
        iterations = G.vcount() - 1

    # Initialize node labels with degree if no iteration limit is set
    # since this is identical to starting at iteration=1
    node_labels = _init_node_labels(G, node_attr, use_degree=iterations is None)

    for _ in range(iterations):

        new_labels = _weisfeiler_lehman_step(G, node_labels, edge_attr=edge_attr)

        prev_labels = node_labels
        node_labels = new_labels

        if node_labels == prev_labels:
            break

    hsh = None
    if return_hash:
        hsh = ";".join(sorted(node_labels))

    if return_graph:
        # Create a new graph with edge labels copied over, and node labels removed
        G = G.copy()
        node_attributes = G.vs.attributes()
        for attr in node_attributes:
            del G.vs[attr]
        G.vs["label"] = node_labels
        return hsh, G
    else:
        return hsh


def _init_k_tuple_labels(
    G: ig.Graph,
    k: int,
    edge_attr: str | List[str] = None,
    node_attr: str | List[str] = None,
) -> List[str]:
    """Initialize the k-tuple labels.

    Note that in contrast with 1-WL that does not consider edges during initialization
    and uses them as part of the update, k-WL test only considers the edges during the
    initialization step.

    Parameters
    ----------
    G : ig.Graph
        The graph.
    k : int
        Degree of the k-WL test, 2 <= k <= 6.
    edge_attr : str | List[str], optional
        The edge attribute(s) to use for the edge labels.
    node_attr : str | List[str], optional
        The node attribute(s) to use for the node labels.

    Returns
    -------
    List[str]
        The node labels.
    """
    if node_attr or edge_attr:
        # Get the canonical permutation of the graph, otherwise
        # edge and node labels cannot be correctly utilized as easily
        # since the order of the edges and the vertices are
        # not preserved in the extracted subgraphs. Basically, we are making
        # hashing easier for colored graphlets, otherwise we would need to use
        # a considerably larger isoclass for each k, considering each different
        # coloring for edges and vertices.
        # See http://www.tcs.hut.fi/Software/bliss/index.html for
        # more information about the BLISS algorithm and canonical permutations.
        G = G.permute_vertices(G.canonical_permutation())

    k_vertex_tuple_generator = itertools.product(range(G.vcount()), repeat=k)
    isoclass = (
        lambda G, node_tuple: G.isoclass(list(node_tuple))
        if k > 2
        else G.are_connected(*node_tuple)
    )

    additional_data = []

    if node_attr:
        initial_node_labels = _init_node_labels(G, node_attr, use_degree=False)
        k_label_tuple_generator = itertools.product(initial_node_labels, repeat=k)
        additional_data.append(k_label_tuple_generator)

    if edge_attr:
        if isinstance(edge_attr, str):
            edge_attr = [edge_attr]
        k_edge_tuples = [
            (
                str(attr)
                for attr_name in edge_attr
                for attr in G.induced_subgraph(node_tuple).es[attr_name]
            )
            for node_tuple in k_vertex_tuple_generator
        ]
        additional_data.append(k_edge_tuples)

    if node_attr or edge_attr:
        k_tuple_labels = [
            ":".join(
                [
                    str(isoclass(G, node_tuple)),
                    *(",;".join(label_tuple) for label_tuple in label_tuples),
                ]
            )
            for node_tuple, label_tuples in zip(
                k_vertex_tuple_generator, zip(additional_data)
            )
        ]
    else:
        k_tuple_labels = [
            str(isoclass(G, node_tuple)) for node_tuple in k_vertex_tuple_generator
        ]

    return k_tuple_labels


def _weisfeiler_lehman_step(
    G: ig.Graph, node_labels: List[str], edge_attr: str | List[str] = None
):
    """Apply neighborhood aggregation to each node in the graph.

    Parameters
    ----------
    G : ig.Graph
        The graph.
    node_labels : List[str]
        The node labels.
    edge_attr : str | List[str], optional
        The edge attribute(s) to use for the edge labels.

    Returns
    -------
    List[str]
        The new node labels.
    """
    return [
        _aggregate_neighborhood(G, node_idx, node_labels, edge_attr=edge_attr)
        for node_idx in range(G.vcount())
    ]


def _k_weisfeiler_lehman_step(
    G: ig.Graph, tuple_labels: List[str], k: int, folklore: bool
):
    """Apply k-Weisfeiler Lehman (k-WL) step.

    Implementations are directly based on the paper [1].

    Parameters
    ----------
    G : ig.Graph
        The graph.
    tuple_labels : List[str]
        The node labels.
    k : int
        Degree of the WL test, 2 <= k <= 6.
    folklore : bool
        Whether to use the folklore variant (k-FWL) of k-WL test.

    Returns
    -------
    List[str]
        The new node labels.

    References
    ----------
    [1] Huang et al., A Short Tutorial on The Weisfeiler-Lehman Test And Its Variants,
        2021. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9413523
    """
    k_vertex_tuple_generator = itertools.product(range(G.vcount()), repeat=k)
    k_tuple_map = dict(zip(k_vertex_tuple_generator, tuple_labels))
    new_labels = []

    if folklore:
        for node_tuple, tuple_label in k_tuple_map.items():
            neighborhood_label_list = [
                ";".join(
                    sorted(
                        [
                            k_tuple_map[node_tuple[:i] + (node,) + node_tuple[i + 1 :]]
                            for i in range(k)
                        ]
                    )
                )
                for node in range(G.vcount())
            ]
            new_label = ":".join(sorted(neighborhood_label_list + [tuple_label]))
            new_labels.append(new_label)

    else:
        for node_tuple, tuple_label in k_tuple_map.items():
            neighborhood_label_list = [
                ";".join(
                    sorted(
                        [
                            k_tuple_map[node_tuple[:i] + (node,) + node_tuple[i + 1 :]]
                            for node in range(G.vcount())
                        ]
                    )
                )
                for i in range(k)
            ]
            new_label = ":".join(sorted(neighborhood_label_list + [tuple_label]))
            new_labels.append(new_label)

    return new_labels


def _init_node_labels(
    G: ig.Graph, node_attr: str | List[str], use_degree: bool = True
) -> List[str]:
    """Initialize the node labels.

    Parameters
    ----------
    G : ig.Graph
        The graph.
    node_attr : str | List[str]
        The node attribute(s) to use as the node labels. If None and
        use_degree, use node degree.
    use_degree : bool
        Whether to use the node degree as part of the node labels.
        By default True.

    Returns
    -------
    List[str]
        The node labels.
    """
    if node_attr:
        if isinstance(node_attr, str):
            node_attr = [node_attr]
        attribute_labels = [G.vs[attr_name] for attr_name in node_attr]
        if use_degree:
            return [
                str(deg) + ":" + ";".join([str(attr) for attr in attributes])
                for attributes, deg in zip(zip(*attribute_labels), G.vs.degree())
            ]
        else:
            return [
                ";".join([str(attr) for attr in attributes])
                for attributes in zip(*attribute_labels)
            ]

    elif use_degree:
        return [str(deg) for deg in G.vs.degree()]

    else:
        return [""] * G.vcount()


def _aggregate_neighborhood(
    G: ig.Graph,
    node_idx: int,
    node_labels: List[str],
    edge_attr: str | List[str] = None,
):
    """Compute new labels for given node by aggregating the labels of each node's neighbors.

    Used for the 1-WL test.

    Parameters
    ----------
    G : ig.Graph
        The graph.
    node_idx : int
        The node index.
    node_labels : List[str]
        The node labels.
    edge_attr : str | List[str], optional
        The edge attribute(s) to use for the edge labels.

    Returns
    -------
    str
        The new label.
    """
    node_neighbors = G.neighbors(node_idx)
    if edge_attr:
        if isinstance(edge_attr, str):
            edge_attr = [edge_attr]
        label_list = [""] * len(node_neighbors)
        for idx, nbr in enumerate(node_neighbors):
            edge = G.es.find(_between=((node_idx,), (nbr,)))
            prefix = ":".join([str(edge[attr]) for attr in edge_attr])
            label_list[idx] = prefix + ";" + node_labels[nbr]
    else:
        label_list = [node_labels[nbr] for nbr in node_neighbors]

    # Add a -1 prefix for consumers of the edge-reduced labels
    # into neat arrays
    label_list = [f"-1;{node_labels[node_idx]}"] + sorted(label_list)
    return ",".join(label_list)


def _reassign_labels(labels_g1, labels_g2=None) -> bool:
    """Reassign the labels of the nodes or node tuples in the graphs.

    Parameters
    ----------
    labels_g1 : List[str]
        The labels of the first graph.
    labels_g2 : List[str], optional
        The labels of the second graph.
    Returns
    -------
    List[str]
        The new labels of the first graph.
    List[str]
        The new labels of the second graph, only present if labels_g2 is not None.
    """
    if labels_g2 is None:
        labels_g2 = []

    all_labels = labels_g1 + labels_g2
    label_map = {label: str(i) for i, label in enumerate(all_labels)}

    new_labels_g1 = [label_map[label] for label in labels_g1]
    new_labels_g2 = [label_map[label] for label in labels_g2]

    if labels_g2:
        return new_labels_g1, new_labels_g2

    return new_labels_g1

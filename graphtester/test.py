"""Use WL to test the graph isomorphism."""
from typing import List

import igraph as ig


def weisfeiler_lehman_test(
    G1: ig.Graph, G2: ig.Graph, edge_attr=None, node_attr=None, iterations=None
) -> bool:
    """Apply 1-Weisfeiler Lehman (1-WL) graph isomorphism test.

    Adapted from networkx.weisfeiler_lehman_graph_hash

    Parameters
    ----------
    G1 : ig.Graph
        The first graph.
    G2 : ig.Graph
        The second graph.
    edge_attr : str, optional
        The edge attribute to use for the edge labels.
    node_attr : str, optional
        The node attribute to use for the node labels.
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

        node_labels_g1, node_labels_g2 = _relabel_nodes(new_labels_g1, new_labels_g2)

        if sorted(node_labels_g1) != sorted(node_labels_g2):
            return False

    return True


def _weisfeiler_lehman_step(G: ig.Graph, node_labels, edge_attr=None):
    """Apply neighborhood aggregation to each node in the graph.

    Parameters
    ----------
    G : ig.Graph
        The graph.
    node_labels : List[str]
        The node labels.
    edge_attr : str, optional
        The edge attribute to use for the edge labels.

    Returns
    -------
    List[str]
        The new node labels.
    """
    return [
        _aggregate_neighborhood(G, node_idx, node_labels, edge_attr=edge_attr)
        for node_idx in range(G.vcount())
    ]


def _init_node_labels(G: ig.Graph, node_attr: str) -> List[str]:
    """Initialize the node labels.

    Parameters
    ----------
    G : ig.Graph
        The graph.
    node_attr : str
        The node attribute to use as the node labels. If None, use node degree.

    Returns
    -------
    List[str]
        The node labels.
    """
    if node_attr:
        return [
            str(attr) + str(deg)
            for attr, deg in zip(G.vs.get_attribute_values(node_attr), G.vs.degree())
        ]
    else:
        return [str(deg) for deg in G.vs.degree()]


def _aggregate_neighborhood(G: ig.Graph, node_idx, node_labels, edge_attr=None):
    """Compute new labels for given node by aggregating the labels of each node's neighbors.

    Parameters
    ----------
    G : ig.Graph
        The graph.
    node_idx : int
        The node index.
    node_labels : List[str]
        The node labels.
    edge_attr : str, optional
        The edge attribute to use for the edge labels.

    Returns
    -------
    str
        The new label.
    """
    if edge_attr is not None:
        label_list = []
        for nbr in G.neighbors(node_idx):
            edge = G.es.find(_between=((node_idx,), (nbr,)))
            prefix = str(edge[edge_attr])
            label_list.append(prefix + node_labels[nbr])
    else:
        label_list = [node_labels[nbr] for nbr in G.neighbors(node_idx)]

    return node_labels[node_idx] + "".join(sorted(label_list))


def _relabel_nodes(node_labels_g1, node_labels_g2) -> bool:
    """Relabel the nodes in the graphs.

    Parameters
    ----------
    node_labels_g1 : List[str]
        The node labels of the first graph.
    node_labels_g2 : List[str]
        The node labels of the second graph.

    Returns
    -------
    List[str]
        The new node labels of the first graph.
    List[str]
        The new node labels of the second graph.
    """
    all_labels = set(node_labels_g1 + node_labels_g2)
    label_map = {label: str(i) for i, label in enumerate(all_labels)}

    new_labels_g1 = [label_map[label] for label in node_labels_g1]
    new_labels_g2 = [label_map[label] for label in node_labels_g2]

    return new_labels_g1, new_labels_g2

"""Playground for substructure counting.

Follows the setting provided by [1]. First estimate the orbits
of the nodes/edges in substructures, via automorphism group
enumeration. Then, label the nodes/edges in candidate graphs
with the orbits of the nodes/edges in substructures, according
to possible isomorphisms.

References
----------
[1] Bouritsas et al., Improving Graph Neural Network Expressivity via
    Subgraph Isomorphism Counting, 2021.
"""
import igraph as ig
import networkx as nx
from matplotlib import pyplot as plt

# These methods are not actually to be used
# externally, but here we do some experimentation
# with them to see if they work as intended.
from graphtester.label import (
    _count_substructure_edges,
    _count_substructure_vertices,
    _determine_edge_orbits,
    _determine_vertex_orbits,
)

# Get a substructure and calculate its vertex orbits
substructure = ig.Graph.Isoclass(4, 6)
plt.figure()
nx.draw(substructure.to_networkx())
print(_determine_vertex_orbits(substructure))
print(substructure.to_networkx().edges())

# Count the isomorphisms of the substructure in a graph
graph = ig.Graph.Isoclass(4, 7)
plt.figure()
nx.draw(
    graph.to_networkx(),
    labels=dict(
        zip(range(graph.vcount()), _count_substructure_vertices(graph, substructure))
    ),
    with_labels=True,
)

# Do the same things with edges
substructure = ig.Graph.Isoclass(4, 6)
plt.figure()
nx.draw(substructure.to_networkx())
print(_determine_edge_orbits(substructure))
print(substructure.to_networkx().edges())

# Count the edge isomorphisms
graph = ig.Graph.Isoclass(4, 7)
plt.figure()
pos = nx.spring_layout(graph.to_networkx())
nx.draw(graph.to_networkx(), pos)
nx.draw_networkx_edge_labels(
    graph,
    pos,
    edge_labels=dict(
        zip(
            [(e.source, e.target) for e in graph.es],
            _count_substructure_edges(graph, substructure),
        )
    ),
)

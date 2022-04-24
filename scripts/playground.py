"""Playground file for the tests."""
import networkx as nx
from matplotlib import pyplot as plt

from graphtester import get_graphs, label_graph
from graphtester import weisfeiler_lehman_test as wl_test

graphs_dict = get_graphs("sr251256")
g1, g2 = graphs_dict[25][:2]
ng1, ng2 = g1.to_networkx(), g2.to_networkx()

# Test (and fail)
print(f"1-WL test: {'failed' if wl_test(g1, g2) else 'succeeded'}")

labeled_g1 = label_graph(g1, ["nbhood_subgraph_comp_sign"])
labeled_g2 = label_graph(g2, ["nbhood_subgraph_comp_sign"])

# Test (and succeed)
labeled_wl_test = wl_test(labeled_g1, labeled_g2, "label", "label")
print(f"1-WL test with signatures: " f"{'failed' if labeled_wl_test else 'succeeded'}")

# Display graphs with signatures as colors
unique_labels = set(labeled_g1.vs["label"] + labeled_g2.vs["label"])
color_map = {
    label: color
    for label, color in zip(unique_labels, ["red", "green", "blue", "cyan"])
}
colors_g1 = [color_map[node] for node in labeled_g1.vs["label"]]
colors_g2 = [color_map[node] for node in labeled_g2.vs["label"]]

plt.figure(figsize=(10, 10))
nx.draw(ng1, pos=nx.shell_layout(ng1), with_labels=True, node_color=colors_g1)
plt.figure(figsize=(10, 10))
nx.draw(ng2, pos=nx.shell_layout(ng2), with_labels=True, node_color=colors_g2)

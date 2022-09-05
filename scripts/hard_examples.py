"""Two hard examples that our approach failed to distinguish."""
import networkx as nx
from matplotlib import pyplot as plt

from graphtester import get_graphs
from graphtester import k_weisfeiler_lehman_test as kwl_test
from graphtester import label_graph
from graphtester import weisfeiler_lehman_test as wl_test

# Dodecahedron:
# https://www.distanceregular.org/graphs/dodecahedron.html
# vs
# Desargues graph ≅ D(O_3):
# https://www.distanceregular.org/graphs/desargues.html
graphs_dict = get_graphs("distance_regular")
g1, g2 = graphs_dict[20][:2]
ng1, ng2 = g1.to_networkx(), g2.to_networkx()

# Test (and fail)
print(f"1-WL test: {'failed' if wl_test(g1, g2) else 'succeeded'}")

# Test (and succeed with 2-FWL)
print(
    f"2-FWL test: {'failed' if kwl_test(g1, g2, k=2, folklore=True) else 'succeeded'}"
)

labeled_g1 = label_graph(g1, ["1st subconstituent signatures"])
labeled_g2 = label_graph(g2, ["1st subconstituent signatures"])
# Test (and again fail)
labeled_wl_test = wl_test(labeled_g1, labeled_g2, "label", "label")
print(f"1-WL test with signatures: " f"{'failed' if labeled_wl_test else 'succeeded'}")

labeled_g1 = label_graph(g1, ["2nd subconstituent signatures"])
labeled_g2 = label_graph(g2, ["2nd subconstituent signatures"])
# Test (and again fail)
labeled_wl_test = wl_test(labeled_g1, labeled_g2, "label", "label")
print(
    f"1-WL test with signatures second subconstitutent: "
    f"{'failed' if labeled_wl_test else 'succeeded'}"
)

# Display graphs with signatures as colors
unique_labels = set(labeled_g1.vs["label"] + labeled_g2.vs["label"])
color_map = {
    label: color for label, color in zip(unique_labels, [f"C{i}" for i in range(10)])
}
colors_g1 = [color_map[node] for node in labeled_g1.vs["label"]]
colors_g2 = [color_map[node] for node in labeled_g2.vs["label"]]

plt.figure(figsize=(10, 10))
nx.draw(ng1, pos=nx.shell_layout(ng1), with_labels=True, node_color=colors_g1)
plt.figure(figsize=(10, 10))
nx.draw(ng2, pos=nx.shell_layout(ng2), with_labels=True, node_color=colors_g2)
plt.show()

# -------------------------------------------------------------------------

# GQ(2,4) minus spread graphs (N=2)
# https://www.distanceregular.org/graphs/gq2.4minusspread.html
# Also see: https://mathworld.wolfram.com/GeneralizedQuadrangle.html
graphs_dict = get_graphs("distance_regular")
g1, g2 = graphs_dict[27][1:3]
ng1, ng2 = g1.to_networkx(), g2.to_networkx()

# Test (and fail)
print(f"1-WL test: {'failed' if wl_test(g1, g2) else 'succeeded'}")

# Test (and fail)
print(
    f"2-FWL test: {'failed' if kwl_test(g1, g2, k=2, folklore=True) else 'succeeded'}"
)

# Test (and succeed with 3-FWL)
print(
    f"3-FWL test: {'failed' if kwl_test(g1, g2, k=3, folklore=True) else 'succeeded'}"
)

labeled_g1 = label_graph(g1, ["1st subconstituent signatures"])
labeled_g2 = label_graph(g2, ["1st subconstituent signatures"])
# Test (and again fail)
labeled_wl_test = wl_test(labeled_g1, labeled_g2, "label", "label")
print(f"1-WL test with signatures: " f"{'failed' if labeled_wl_test else 'succeeded'}")

labeled_g1 = label_graph(g1, ["2nd subconstituent signatures"])
labeled_g2 = label_graph(g2, ["2nd subconstituent signatures"])
# Test (and again fail)
labeled_wl_test = wl_test(labeled_g1, labeled_g2, "label", "label")
print(
    f"1-WL test with signatures second subconstitutent: "
    f"{'failed' if labeled_wl_test else 'succeeded'}"
)

# Display graphs with signatures as colors
unique_labels = set(labeled_g1.vs["label"] + labeled_g2.vs["label"])
color_map = {
    label: color for label, color in zip(unique_labels, [f"C{i}" for i in range(10)])
}
colors_g1 = [color_map[node] for node in labeled_g1.vs["label"]]
colors_g2 = [color_map[node] for node in labeled_g2.vs["label"]]

plt.figure(figsize=(10, 10))
nx.draw(ng1, pos=nx.shell_layout(ng1), with_labels=True, node_color=colors_g1)
plt.figure(figsize=(10, 10))
nx.draw(ng2, pos=nx.shell_layout(ng2), with_labels=True, node_color=colors_g2)
plt.show()

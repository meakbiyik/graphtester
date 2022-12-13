"""Playground file for the WL tests."""
import networkx as nx
from matplotlib import pyplot as plt

from graphtester import k_weisfeiler_lehman_test as kwl_test
from graphtester import label, produce
from graphtester import weisfeiler_lehman_test as wl_test

graphs_dict = produce("strongly_regular")
g1, g2 = graphs_dict[16][:2]
ng1, ng2 = g1.to_networkx(), g2.to_networkx()

# Test (and fail)
print(f"1-WL test: {'failed' if wl_test(g1, g2) else 'succeeded'}")

# Test (and again fail as k-WL with k=2 is equivalent to 1-WL above)
print(f"2-WL test: {'failed' if kwl_test(g1, g2, k=2) else 'succeeded'}")

# Test (and again fail with 3-WL)
print(f"3-WL test: {'failed' if kwl_test(g1, g2, k=3) else 'succeeded'}")

# Test (and again fail with 2-FWL, since it is equivalent to 3-WL above)
print(
    f"2-FWL test: {'failed' if kwl_test(g1, g2, k=2, folklore=True) else 'succeeded'}"
)

# Test (and succeed with 4-WL)
print(f"4-WL test: {'failed' if kwl_test(g1, g2, k=4) else 'succeeded'}")

# Test (and succeed with 3-FWL, since it is equivalent to 4-WL above)
print(
    f"3-FWL test: {'failed' if kwl_test(g1, g2, k=3, folklore=True) else 'succeeded'}"
)

labeled_g1 = label(g1, ["1st subconstituent signatures"])
labeled_g2 = label(g2, ["1st subconstituent signatures"])

# Test (and succeed)
labeled_wl_test = wl_test(labeled_g1, labeled_g2, "label", "label")
print(f"1-WL test with signatures: " f"{'failed' if labeled_wl_test else 'succeeded'}")

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

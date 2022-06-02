"""Compare the performances of different labeling methods."""
import timeit
from pathlib import Path

import pandas as pd
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt

from graphtester import GRAPH_CLASS_DESCRIPTIONS, get_graphs, label_graph

RESULTS_DIR = Path(__file__).parents[1] / "results"

classes = ["all", "highlyirregular", "strongly_regular"]
methods = [
    ("nbhood_subgraph_comp_sign", "edge_betweenness"),
    ("4_path_count_edge",),
    ("5_path_count_edge",),
    ("6_path_count_edge",),
    ("4_clique_count_edge",),
    ("5_clique_count_edge",),
    ("6_clique_count_edge",),
]
times = []

for cls in classes:
    graphs_dict = get_graphs(cls, max_node_count=30)

    for node_count, graphs in tqdm.tqdm(graphs_dict.items()):
        for graph in graphs:
            for method in methods:
                time = timeit.timeit(lambda: label_graph(graph, method), number=10) / 10
                times.append(
                    {
                        "graph_class": GRAPH_CLASS_DESCRIPTIONS[cls],
                        "node_count": node_count,
                        "labeling": " + ".join(method),
                        "time": time,
                    }
                )

times_df = pd.DataFrame(times)

plt.figure(figsize=(12, 12))
g = sns.lineplot(
    x="node_count",
    y="time",
    hue="labeling",
    style="graph_class",
    data=times_df,
    err_style="bars",
)
g.set_yscale("log")
plt.savefig(RESULTS_DIR / "labeling_methods_timings.pdf")

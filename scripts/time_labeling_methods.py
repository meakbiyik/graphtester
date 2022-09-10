"""Compare the performances of different labeling methods."""
import timeit
from pathlib import Path

import pandas as pd
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt

from graphtester import GRAPH_CLASS_DESCRIPTIONS, get_graphs, label_graph

RESULTS_DIR = Path(__file__).parents[1] / "results"
sns.set_context("paper", font_scale=1.5)
sns.set_theme(style="whitegrid")

classes = ["strongly_regular", "all", "highlyirregular"]
methods = [
    ("1st subconstituent signatures", "Edge betweenness"),
    ("2nd subconstituent signatures", "Edge betweenness"),
    ("4-clique count of edges",),
    ("5-path count of edges",),
    ("6-path count of vertices",),
    ("6-cycle count of vertices",),
]
times = []

for cls in classes:
    graphs_dict = get_graphs(cls, max_node_count=40)

    for node_count, graphs in tqdm.tqdm(graphs_dict.items()):
        rep = 10 if node_count < 20 else 3
        count = 1000 if node_count < 20 else 50
        for graph in graphs[:count]:
            for method in methods:
                time = (
                    timeit.timeit(lambda: label_graph(graph, method), number=rep) / rep
                )
                times.append(
                    {
                        "Graph class": GRAPH_CLASS_DESCRIPTIONS[cls],
                        "node_count": node_count,
                        "Method": " + ".join(method),
                        "time": time,
                    }
                )

times_df = pd.DataFrame(times)

plt.figure(figsize=(16, 12))
g = sns.lineplot(
    x="node_count",
    y="time",
    hue="Method",
    style="Graph class",
    data=times_df,
    err_style="bars",
    linewidth=1.5,
)
g.set_yscale("log")
plt.xlabel("Number of nodes")
plt.ylabel("Time (s)")
plt.grid(True, which="major", axis="y")
plt.savefig(RESULTS_DIR / "labeling_methods_timings.pdf")

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

classes = ["all", "highlyirregular", "strongly_regular"]
methods = [
    ("1st subconstituent signatures", "Edge betweenness"),
    ("2nd subconstituent signatures", "Edge betweenness"),
    ("4-path count of edges",),
    ("5-path count of edges",),
    ("6-path count of edges",),
    ("4-clique count of edges",),
    ("5-clique count of edges",),
    ("6-clique count of edges",),
]
times = []

for cls in classes:
    graphs_dict = get_graphs(cls, max_node_count=40)

    for node_count, graphs in tqdm.tqdm(graphs_dict.items()):
        for graph in graphs[:200]:
            for method in methods:
                time = timeit.timeit(lambda: label_graph(graph, method), number=10) / 10
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
    linewidth=3,
)
g.set_yscale("log")
plt.xlabel("Number of nodes")
plt.ylabel("Time (s)")
plt.grid(True, which="major", axis="y")
plt.savefig(RESULTS_DIR / "labeling_methods_timings.pdf")

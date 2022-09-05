"""Create barplots to depict the performance of different pre-coloring methods."""
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

RESULTS_DIR = Path(__file__).parents[1] / "results"
sns.set_context("paper", font_scale=2)
sns.set_theme(style="whitegrid")

results = pd.DataFrame(
    [
        {
            "Method": "3-path vertex count",
            "Method class": "Substructure counting - path",
            "Failure count": 54403,
        },
        {
            "Method": "4-path vertex count",
            "Method class": "Substructure counting - path",
            "Failure count": 23158,
        },
        {
            "Method": "5-path vertex count",
            "Method class": "Substructure counting - path",
            "Failure count": 18077,
        },
        {
            "Method": "6-path vertex count",
            "Method class": "Substructure counting - path",
            "Failure count": 17688,
        },
        {
            "Method": "3-path edge count",
            "Method class": "Substructure counting - path",
            "Failure count": 53825,
        },
        {
            "Method": "4-path edge count",
            "Method class": "Substructure counting - path",
            "Failure count": 18094,
        },
        {
            "Method": "5-path edge count",
            "Method class": "Substructure counting - path",
            "Failure count": 17685,
        },
        {
            "Method": "6-path edge count",
            "Method class": "Substructure counting - path",
            "Failure count": 17688,
        },
        {
            "Method": "3-cycle vertex count",
            "Method class": "Substructure counting - cycle",
            "Failure count": 23158,
        },
        {
            "Method": "4-cycle vertex count",
            "Method class": "Substructure counting - cycle",
            "Failure count": 18199,
        },
        {
            "Method": "5-cycle vertex count",
            "Method class": "Substructure counting - cycle",
            "Failure count": 17906,
        },
        {
            "Method": "6-cycle vertex count",
            "Method class": "Substructure counting - cycle",
            "Failure count": 17734,
        },
        {
            "Method": "3-cycle edge count",
            "Method class": "Substructure counting - cycle",
            "Failure count": 18102,
        },
        {
            "Method": "4-cycle edge count",
            "Method class": "Substructure counting - cycle",
            "Failure count": 17800,
        },
        {
            "Method": "5-cycle edge count",
            "Method class": "Substructure counting - cycle",
            "Failure count": 17766,
        },
        {
            "Method": "6-cycle edge count",
            "Method class": "Substructure counting - cycle",
            "Failure count": 17706,
        },
        {
            "Method": "3-clique vertex count",
            "Method class": "Substructure counting - clique",
            "Failure count": 23158,
        },
        {
            "Method": "4-clique vertex count",
            "Method class": "Substructure counting - clique",
            "Failure count": 7736,
        },
        {
            "Method": "5-clique vertex count",
            "Method class": "Substructure counting - clique",
            "Failure count": 52849,
        },
        {
            "Method": "6-clique vertex count",
            "Method class": "Substructure counting - clique",
            "Failure count": 54356,
        },
        {
            "Method": "3-clique edge count",
            "Method class": "Substructure counting - clique",
            "Failure count": 18102,
        },
        {
            "Method": "4-clique edge count",
            "Method class": "Substructure counting - clique",
            "Failure count": 6021,
        },
        {
            "Method": "5-clique edge count",
            "Method class": "Substructure counting - clique",
            "Failure count": 52826,
        },
        {
            "Method": "6-clique edge count",
            "Method class": "Substructure counting - clique",
            "Failure count": 54354,
        },
        {
            "Method": "Edge betweenness",
            "Method class": "Ours",
            "Failure count": 17692,
        },
        {
            "Method": "1st subconstituent signatures",
            "Method class": "Ours",
            "Failure count": 98,
        },
        {
            "Method": "2nd subconstituent signatures",
            "Method class": "Ours",
            "Failure count": 111,
        },
        {
            "Method": "1st subconstituent signatures + Edge betweenness",
            "Method class": "Ours",
            "Failure count": 14,
        },
        {
            "Method": "2nd subconstituent signatures + Edge betweenness",
            "Method class": "Ours",
            "Failure count": 8,
        },
        {
            "Method": "3-WL",
            "Method class": "k-WL",
            "Failure count": 17684,
        },
        {
            "Method": "4-WL",
            "Method class": "k-WL",
            "Failure count": 0,
        },
    ]
)

results["Success rate"] = (54403 - results["Failure count"]) / 54403

fig, ax = plt.subplots(figsize=(12, 16))
sns.barplot(
    x="Method",
    y="Success rate",
    hue="Method class",
    data=results,
    ax=ax,
    dodge=False,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel("Success rate")
ax.set_xlabel("Method")
ax.legend()
plt.grid(True, which="major", axis="y")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "precoloring_success_rate.pdf")

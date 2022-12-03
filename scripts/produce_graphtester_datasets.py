"""Produce a versioned graphtester datasets (full and small)."""

import json
import sys
from pathlib import Path

from graphtester import FAST_GRAPH_CLASSES, GRAPH_CLASSES, load, produce

VERSION = "0.1.0"

if __name__ == "__main__":

    DATASET_FOLDER = Path(__file__).parents[1] / "graphtester" / "io" / "datasets"

    # Check the version of the datasets if they already exist
    if (DATASET_FOLDER / "GT.json").exists():
        with open(DATASET_FOLDER / "GT.json", "r") as f:
            version = json.load(f)["version"]
        if version == VERSION:
            print("Dataset already exists and is up to date.")
            sys.exit(1)

    # Remove old datasets
    for file in DATASET_FOLDER.glob("GT*"):
        file.unlink()

    classes_full = {
        graph_class: [n for n in node_counts if n <= 40]
        for graph_class, node_counts in GRAPH_CLASSES.items()
    }
    graphs_full = {
        cls: produce(cls, max(node_counts)) for cls, node_counts in classes_full.items()
    }

    classes_small = {
        graph_class: [n for n in node_counts if n <= 20]
        for graph_class, node_counts in FAST_GRAPH_CLASSES.items()
    }
    graphs_small = {
        cls: produce(cls, max(node_counts))
        for cls, node_counts in classes_small.items()
    }

    for graphs, classes, suffix in [
        (graphs_full, classes_full, ""),
        (graphs_small, classes_small, "-small"),
    ]:
        name = f"GT{suffix}"
        flat_graphs = [
            graph
            for graphs_dict in graphs.values()
            for graphs in graphs_dict.values()
            for graph in graphs
        ]
        dataset = load(flat_graphs, dataset_name=f"{name} (Version: {VERSION})")
        dataset.to_pickle(DATASET_FOLDER / f"{name}.pkl")

        # Also create a JSON file with the information on the datasets,
        # e.g., the names of classes in dataset, number of graphs per class, etc.
        dataset_info = {
            "name": name,
            "version": VERSION,
            "classes": list(graphs.keys()),
            "node_counts": classes,
            "num_graphs": {
                cls: sum([len(g) for g in graphs_dict.values()])
                for cls, graphs_dict in graphs.items()
            },
        }
        with open(DATASET_FOLDER / f"{name}.json", "w") as f:
            json.dump(dataset_info, f, indent=4)

    print("Done.")

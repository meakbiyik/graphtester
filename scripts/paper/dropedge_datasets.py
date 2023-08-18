"""Tests on synthetic dataset implementations.

Directly from DropGNN paper.
"""
from typing import Type
import numpy as np
import networkx as nx
import pandas as pd
from torch_geometric.utils.convert import from_networkx, to_networkx
from torch_geometric.data import Data
import torch

import graphtester as gt
from graphtester.evaluate.dataset import DEFAULT_METRICS


# Synthetic datasets
class SymmetrySet:
    def __init__(self):
        self.hidden_units = 0
        self.num_classes = 0
        self.num_features = 0
        self.num_nodes = 0

    def create_dataset(self):
        pass


class LimitsOne(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def create_dataset(self):
        n_nodes = 16  # There are two connected components, each with 8 nodes

        colors = [0, 1, 2, 3] * 4
        y = torch.tensor([0] * 8 + [1] * 8)
        edge_index = torch.tensor(
            [
                [
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    0,
                    4,
                    5,
                    5,
                    6,
                    6,
                    7,
                    7,
                    4,
                    8,
                    9,
                    9,
                    10,
                    10,
                    11,
                    11,
                    12,
                    12,
                    13,
                    13,
                    14,
                    14,
                    15,
                    15,
                    8,
                ],
                [
                    1,
                    0,
                    2,
                    1,
                    3,
                    2,
                    0,
                    3,
                    5,
                    4,
                    6,
                    5,
                    7,
                    6,
                    4,
                    7,
                    9,
                    8,
                    10,
                    9,
                    11,
                    10,
                    12,
                    11,
                    13,
                    12,
                    14,
                    13,
                    15,
                    14,
                    8,
                    15,
                ],
            ],
            dtype=torch.long,
        )
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        subgraph1, subgraph2 = data.subgraph(
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        ), data.subgraph(torch.tensor([8, 9, 10, 11, 12, 13, 14, 15]))
        nx_graph1, nx_graph2 = to_networkx(subgraph1, to_undirected=True), to_networkx(
            subgraph2, to_undirected=True
        )
        return gt.load(
            name_or_graphs=[nx_graph1, nx_graph2],
            node_labels=[y[:8], y[8:]],
            dataset_name="limits1",
        )


class LimitsTwo(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def create_dataset(self):
        n_nodes = 16  # There are two connected components, each with 8 nodes

        colors = [0, 1, 2, 3] * 4
        y = torch.tensor([0] * 8 + [1] * 8)
        edge_index = torch.tensor(
            [
                [
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    0,
                    4,
                    5,
                    5,
                    6,
                    6,
                    7,
                    7,
                    4,
                    1,
                    3,
                    5,
                    7,
                    8,
                    9,
                    9,
                    10,
                    10,
                    11,
                    11,
                    8,
                    12,
                    13,
                    13,
                    14,
                    14,
                    15,
                    15,
                    12,
                    9,
                    15,
                    11,
                    13,
                ],
                [
                    1,
                    0,
                    2,
                    1,
                    3,
                    2,
                    0,
                    3,
                    5,
                    4,
                    6,
                    5,
                    7,
                    6,
                    4,
                    7,
                    3,
                    1,
                    7,
                    5,
                    9,
                    8,
                    10,
                    9,
                    11,
                    10,
                    8,
                    11,
                    13,
                    12,
                    14,
                    13,
                    15,
                    14,
                    12,
                    15,
                    15,
                    9,
                    13,
                    11,
                ],
            ],
            dtype=torch.long,
        )
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        subgraph1, subgraph2 = data.subgraph(
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        ), data.subgraph(torch.tensor([8, 9, 10, 11, 12, 13, 14, 15]))
        nx_graph1, nx_graph2 = to_networkx(subgraph1), to_networkx(subgraph2)
        return gt.load(
            name_or_graphs=[nx_graph1, nx_graph2],
            node_labels=[y[:8], y[8:]],
            dataset_name="limits2",
        )


class Triangles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 60
        self.graph_class = False

    def create_dataset(self):
        size = self.num_nodes
        generated = False
        while not generated:
            nx_g = nx.random_degree_sequence_graph([3] * size)
            data = from_networkx(nx_g)
            labels = [0] * size
            for n in range(size):
                for nb1 in data.edge_index[1][data.edge_index[0] == n]:
                    for nb2 in data.edge_index[1][data.edge_index[0] == n]:
                        if torch.logical_and(
                            data.edge_index[0] == nb1, data.edge_index[1] == nb2
                        ).any():
                            labels[n] = 1
            generated = labels.count(0) >= 20 and labels.count(1) >= 20
        data.y = np.array(labels)

        nx_graph = to_networkx(data)

        return gt.load(
            name_or_graphs=[nx_graph],
            node_labels=[data.y],
            dataset_name="triangles",
        )


class LCC(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 3
        self.num_features = 1
        self.num_nodes = 10
        self.graph_class = False

    def create_dataset(self):
        generated = False
        while not generated:
            graphs = []
            labels = []
            i = 0
            while i < 6:
                size = 10
                nx_g = nx.random_degree_sequence_graph([3] * size)
                if nx.is_connected(nx_g):
                    i += 1
                    data = from_networkx(nx_g)
                    lbls = [0] * size
                    for n in range(size):
                        edges = 0
                        nbs = [
                            int(nb)
                            for nb in data.edge_index[1][data.edge_index[0] == n]
                        ]
                        for nb1 in nbs:
                            for nb2 in nbs:
                                if torch.logical_and(
                                    data.edge_index[0] == nb1, data.edge_index[1] == nb2
                                ).any():
                                    edges += 1
                        lbls[n] = int(edges / 2)
                    data.y = np.array(lbls)
                    labels.extend(lbls)
                    graphs.append(data)
            generated = (
                labels.count(0) >= 10
                and labels.count(1) >= 10
                and labels.count(2) >= 10
            )  # Ensure the dataset is somewhat balanced

        nx_graphs = [to_networkx(g) for g in graphs]

        return gt.load(
            name_or_graphs=nx_graphs,
            node_labels=[g.y for g in graphs],
            dataset_name="lcc",
        )


class FourCycles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.p = 4
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 4 * self.p
        self.graph_class = True

    def gen_graph(self, p):
        edge_index = None
        for i in range(p):
            e = torch.tensor(
                [[i, p + i, 2 * p + i, 3 * p + i], [2 * p + i, 3 * p + i, i, p + i]],
                dtype=torch.long,
            )
            if edge_index is None:
                edge_index = e
            else:
                edge_index = torch.cat([edge_index, e], dim=-1)
        top = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            top[i * p + t] = 1
        bottom = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            bottom[i * p + t] = 1
        for i, bit in enumerate(top):
            if bit:
                e = torch.tensor(
                    [[i // p, p + i % p], [p + i % p, i // p]], dtype=torch.long
                )
                edge_index = torch.cat([edge_index, e], dim=-1)
        for i, bit in enumerate(bottom):
            if bit:
                e = torch.tensor(
                    [[2 * p + i // p, 3 * p + i % p], [3 * p + i % p, 2 * p + i // p]],
                    dtype=torch.long,
                )
                edge_index = torch.cat([edge_index, e], dim=-1)
        return Data(edge_index=edge_index, num_nodes=self.num_nodes), any(
            np.logical_and(top, bottom)
        )

    def create_dataset(self):
        size = 25
        p = self.p
        trues = []
        falses = []
        while len(trues) < size or len(falses) < size:
            data, label = self.gen_graph(p)
            data.y = label
            nx_graph = to_networkx(data)
            if label and len(trues) < size:
                trues.append(nx_graph)
            elif not label and len(falses) < size:
                falses.append(nx_graph)
        return gt.load(
            name_or_graphs=trues + falses,
            labels=[1] * size + [0] * size,
            dataset_name="fourcycles",
        )


class SkipCircles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 32
        self.num_classes = 10  # num skips
        self.num_features = 1
        self.num_nodes = 41
        self.graph_class = True

    def create_dataset(self):
        size = self.num_nodes
        skips = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
        graphs = []
        for s, skip in enumerate(skips):
            edge_index = torch.tensor([[0, size - 1], [size - 1, 0]], dtype=torch.long)
            for i in range(size - 1):
                e = torch.tensor([[i, i + 1], [i + 1, i]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
            for i in range(size):
                e = torch.tensor(
                    [[i, i], [(i - skip) % size, (i + skip) % size]], dtype=torch.long
                )
                edge_index = torch.cat([edge_index, e], dim=-1)
            data = Data(edge_index=edge_index, num_nodes=self.num_nodes)
            data.y = s
            graphs.append(data)

        nx_graphs = [to_networkx(g) for g in graphs]
        return gt.load(
            name_or_graphs=nx_graphs,
            labels=[g.y for g in graphs],
            dataset_name="skipcircles",
        )


# Test them with graphtester. Limits1 and Limits2 are deterministic but the others
# are random, we test them 10 times and report the average and std.
metrics = {
    "LimitsOne": DEFAULT_METRICS[f"upper_bound_accuracy_node"],
    "LimitsTwo": DEFAULT_METRICS[f"upper_bound_accuracy_node"],
    "Triangles": DEFAULT_METRICS[f"upper_bound_accuracy_node"],
    "LCC": DEFAULT_METRICS[f"upper_bound_accuracy_node"],
    "FourCycles": DEFAULT_METRICS[f"upper_bound_accuracy"],
    "SkipCircles": DEFAULT_METRICS[f"upper_bound_accuracy"],
}


def evaluate(GraphClass: Type[SymmetrySet], use_transformer_basis, repeats=10):
    """
    evaluation = gt.evaluate(
        dataset,
        metrics=metrics,
        ignore_node_features=True,
        ignore_edge_features=True,
        iterations=3,
        transformer=use_transformer_basis,
    )
    print(f"Evaluation for {dataset.name} with transformer={use_transformer_basis}")
    print(evaluation, flush=True)
    """
    results = []
    for i in range(repeats):
        dataset = GraphClass().create_dataset()
        evaluation = gt.evaluate(
            dataset,
            metrics=[metrics[GraphClass.__name__]],
            ignore_node_features=False,
            ignore_edge_features=True,
            iterations=3,
            transformer=use_transformer_basis,
        )
        eval_df = evaluation.as_dataframe()
        results.append(eval_df)
    print(f"Average for {dataset.name} with transformer={use_transformer_basis}")
    print(pd.concat(results).groupby(level=0).mean(), flush=True)
    if repeats > 1:
        print(f"Std for {dataset.name} with transformer={use_transformer_basis}")
        print(pd.concat(results).groupby(level=0).std(), flush=True)


if __name__ == "__main__":
    evaluate(LimitsOne, False, repeats=1)
    evaluate(LimitsTwo, False, repeats=1)
    evaluate(LCC, False, repeats=10)
    evaluate(Triangles, False, repeats=10)
    evaluate(FourCycles, False, repeats=10)
    evaluate(SkipCircles, False, repeats=1)
    # for node classificaiton, also try out transformer basis
    evaluate(LimitsOne, True, repeats=1)
    evaluate(LimitsTwo, True, repeats=1)
    evaluate(Triangles, True, repeats=10)
    evaluate(LCC, True, repeats=10)

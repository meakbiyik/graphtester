"""Experiments with GNNs on MUTAG dataset."""
import shutil
from pathlib import Path

from model import GNN
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv  # noqa: F401
from train import train

import graphtester as gt

SCRIPT_PATH = Path(__file__).parent

# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

dataset = TUDataset(
    root=SCRIPT_PATH / "data" / "MUTAG",
    name="MUTAG",
    pre_transform=gt.pretransform(
        features=["Closeness centrality", "Eigenvector centrality"],
        feature_names="x",
    ),
)

print(dataset[0].x)
print(dataset[0].x.shape)
node_feature_count = dataset[0].x.shape[1]
num_classes = dataset.num_classes

model = GNN(node_feature_count, num_classes, 20, 0.1, conv=GINConv)
train(dataset, model)

# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

from model import GNN
from train import train

from torch_geometric.nn import GINConv, GCNConv, GATv2Conv

import shutil
import torch
from pathlib import Path

from torch_geometric.datasets import TUDataset

import graphtester as gt

SCRIPT_PATH = Path(__file__).parent

# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

dataset = TUDataset(
    root=SCRIPT_PATH / "data" / "MUTAG",
    name="MUTAG",
    pre_transform=gt.pretransform(
        features=["Closeness centrality", "Eigenvector centrality"],
        encode=True,
        encode_together=True,
    ),
)

print(dataset[0].x)
print(dataset[0]["_encoding"])


model = GNN(8, 2, 20, 0.1, conv=GINConv)
train(dataset, model)






# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

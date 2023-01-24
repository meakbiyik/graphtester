"""Experiments on torch-geometric with pretransform."""
import shutil
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
        features=["Edge betweenness", "Eigenvector centrality"]
    ),
)

print(dataset[0])
print(dataset[0]["_edge_betweenness"])
print(dataset[0]["_eigenvector_centrality"])

# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

dataset = TUDataset(
    root=SCRIPT_PATH / "data" / "MUTAG",
    name="MUTAG",
    pre_transform=gt.pretransform(
        features=["Closeness centrality", "Eigenvector centrality"],
        feature_names="feature",
    ),
)

print(dataset[0])
print(dataset[0].feature.shape)

# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

dataset = TUDataset(
    root=SCRIPT_PATH / "data" / "MUTAG",
    name="MUTAG",
    pre_transform=gt.pretransform(
        features=["Closeness centrality", "Eigenvector centrality"], feature_names="x"
    ),
)

print(dataset[0])
print(dataset[0].x.shape)

# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

dataset = TUDataset(
    root=SCRIPT_PATH / "data" / "MUTAG",
    name="MUTAG",
    pre_transform=gt.pretransform(
        features=["Closeness centrality", "Eigenvector centrality"],
        encode=True,
    ),
)

print(dataset[0])
print(dataset[0]["_closeness_centrality"])
print(dataset[0]["_eigenvector_centrality"])
print(dataset[1]["_closeness_centrality"])
print(dataset[1]["_eigenvector_centrality"])

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

print(dataset[0])
print(dataset[0]["_encoding"])
print(dataset[1]["_encoding"])

# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

try:
    dataset = TUDataset(
        root=SCRIPT_PATH / "data" / "MUTAG",
        name="MUTAG",
        pre_transform=gt.pretransform(
            features=["Closeness centrality", "1st subconstituent signatures"],
            encode=False,
        ),
    )
except ValueError as e:
    print("Expected ValueError:")
    print(e)

# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

dataset = TUDataset(
    root=SCRIPT_PATH / "data" / "MUTAG",
    name="MUTAG",
    pre_transform=gt.pretransform(
        features=["Closeness centrality", "1st subconstituent signatures"],
        encode=True,
        encode_together=True,
    ),
)

print(dataset[0])
print(dataset[0]["_encoding"])

# remove the pretransformed dataset from the folder
shutil.rmtree(SCRIPT_PATH / "data" / "MUTAG", ignore_errors=True)

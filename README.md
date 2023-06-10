# Graphtester

Graphtester is a Python package for comprehensive analysis of the theoretical capabilities of GNNs for various datasets, tasks, and scores, using and extension of the Weisfeiler-Lehman framework.

## Usage

Graphtester can load, label, analyze, and test datasets. The following example shows how to load a dataset, label it, and test it with a GNN.

### Loading a dataset

Graphtester package exposes a `load` function that can load a dataset from various formats, and convert it to a `Dataset` object, internal storage format of Graphtester. `load` function can take a dataset name, list of NetworkX or iGraph graphs, or a PyG/DGL dataset as input. The following example loads the `MUTAG` dataset.

```python
import graphtester as gt

dataset = gt.load("MUTAG")
```
<details>
<summary>See datasets that can be loaded with their names</summary>

| Name                  | Description                                           | Task                 |
|-----------------------|-------------------------------------------------------|----------------------|
| `GT`                  | Synthetic Graphtester dataset for benchmarking labels | -                    |
| `GT-small`            | Smaller version of GT                                 | -                    |
| `ZINC_FULL`           | Full ZINC dataset                                     | Graph classification |
| `ZINC`                | Subset of ZINC                                        | Graph classification |
| `MNIST`               | MNIST dataset                                         | Graph classification |
| `CIFAR10`             | CIFAR10 dataset                                       | Graph classification |
| `PATTERN`             | PATTERN dataset                                       | Node classification  |
| `CLUSTER`             | CLUSTER dataset                                       | Node classification  |
| `AIDS`                | AIDS dataset                                          | Graph classification |
| `BZR`                 | BZR dataset                                           | Graph classification |
| `BZR_MD`              | BZR_MD dataset                                        | Graph classification |
| `COX2`                | COX2 dataset                                          | Graph classification |
| `COX2_MD`             | COX2_MD dataset                                       | Graph classification |
| `DHFR`                | DHFR dataset                                          | Graph classification |
| `DHFR_MD`             | DHFR_MD dataset                                       | Graph classification |
| `ER_MD`               | ER_MD dataset                                         | Graph classification |
| `FRANKENSTEIN`        | FRANKENSTEIN dataset                                  | Graph classification |
| `Mutagenicity`        | Mutagenicity dataset                                  | Graph classification |
| `MUTAG`               | MUTAG dataset                                         | Graph classification |
| `NCI1`                | NCI1 dataset                                          | Graph classification |
| `NCI109`              | NCI109 dataset                                        | Graph classification |
| `PTC_FM`              | PTC_FM dataset                                        | Graph classification |
| `PTC_FR`              | PTC_FR dataset                                        | Graph classification |
| `PTC_MM`              | PTC_MM dataset                                        | Graph classification |
| `PTC_MR`              | PTC_MR dataset                                        | Graph classification |
| `ENZYMES`             | ENZYMES dataset                                       | Graph classification |
| `DD`                  | DD dataset                                            | Graph classification |
| `PROTEINS`            | PROTEINS dataset                                      | Graph classification |
| `Fingerprint`         | Fingerprint dataset                                   | Graph classification |
| `Cuneiform`           | Cuneiform dataset                                     | Graph classification |
| `COIL-DEL`            | COIL-DEL dataset                                      | Graph classification |
| `COIL-RAG`            | COIL-RAG dataset                                      | Graph classification |
| `MSRC_9`              | MSRC_9 dataset                                        | Graph classification |
| `IMDB-BINARY`         | IMDB-BINARY dataset                                   | Graph classification |
| `IMDB-MULTI`          | IMDB-MULTI dataset                                    | Graph classification |
| `COLLAB`              | COLLAB dataset                                        | Graph classification |
| `REDDIT-BINARY`       | REDDIT-BINARY dataset                                 | Graph classification |
| `REDDIT-MULTI-5K`     | REDDIT-MULTI-5K dataset                               | Graph classification |
| `ogbg-molbbbp`        | OGBG-MOLBBBP dataset                                  | Graph classification |
| `ogbg-molhiv`         | OGBG-MOLHIV dataset                                   | Graph classification |
| `ogbn-arxiv`          | OGBN-ARXIV dataset                                    | Node classification  |
| `ogbn-proteins`       | OGBN-PROTEINS dataset                                 | Node classification  |
| `ogbn-products`       | OGBN-PRODUCTS dataset                                 | Node classification  |
| `ogbg-molesol`        | OGBG-MOLESOL dataset                                  | Graph regression     |
| `ogbg-molfreesolv`    | OGBG-MOLFREESOLV dataset                              | Graph regression     |
| `ogbg-mollipo`        | OGBG-MOLLIPO dataset                                  | Graph regression     |
| `Cora`                | Cora dataset                                          | Node classification  |
| `Citeseer`            | Citeseer dataset                                      | Node classification  |
| `Pubmed`              | Pubmed dataset                                        | Node classification  |
| `CoauthorCS`          | CoauthorCS dataset                                    | Node classification  |
| `CoauthorPhysics`     | CoauthorPhysics dataset                               | Node classification  |
| `AmazonCoBuyComputer` | AmazonCoBuyComputer dataset                           | Node classification  |
</details>

### Running 1-WL or k-WL on a graph pair

Graphtester can run 1-WL or k-WL on a pair of graphs. The following example runs 1-WL on the first two graphs in the dataset.

```python
from graphtester import (
    weisfeiler_lehman_test as wl_test,
    k_weisfeiler_lehman_test as kwl_test
)

G1, G2 = dataset.graphs[:2]
is_iso = wl_test(G1, G2)
is_iso_kwl = kwl_test(G1, G2, k=4)
```

### Computing upper score bounds for a dataset

Graphtester can compute upper score bounds for a dataset and task. For calculating the upper score bounds associated with a specified number of layers, we utilize the congruence of Graph Neural Networks (GNNs) and graph transformers with the 1-Weisfeiler-Lehman test (1-WLE), as established in the paper.

```python
import graphtester as gt

dataset = gt.load("ZINC")
evaluation = gt.evaluate(dataset)
print(evaluation.as_dataframe())
```

## Installation

Simply run

```bash
pip install git+https://github.com/meakbiyik/graphtester.git
```

## Reproducing the results in the paper

1. Install the package
2. Run `sbatch scripts/paper/test_all_datasets.sh` to run all experiments on the datasets. Note that this scripts requires 10 cores and substantial amount of memory. You can change the number of cores by changing the `--cpus-per-task` argument in the script.
3. Recommendation results can be gathered by adjusting the parameters in `scripts/paper/test_all_datasets.py` script for the desired task and running the bash script again.

## Development

1. Install [`poetry`](https://python-poetry.org/docs/#installation)
2. Run `poetry install` to install dependencies
3. Run `poetry run pre-commit install` to install pre-commit hooks

### Environment

You might want to set the following environment variables:

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring # disable keyring, might be needed for poetry
export SCRATCH=/itet-stor/${USER}/net_scratch # set cache locations
export XDG_CACHE_HOME=$SCRATCH/.cache # generic Linux cache
export POETRY_VIRTUALENVS_PATH=$SCRATCH/envs # poetry environment
export GRAPHTESTER_CACHE_DIR=$SCRATCH/.graphtester # graphtester cache
```

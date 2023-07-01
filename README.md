# Graphtester

[![Lint and Test](https://github.com/meakbiyik/graphtester/actions/workflows/package.yaml/badge.svg)](https://github.com/meakbiyik/graphtester/actions/workflows/package.yaml) [![Documentation Status](https://readthedocs.org/projects/graphtester/badge/?version=latest)](https://graphtester.readthedocs.io/en/latest/?badge=latest)

Graphtester is a Python package for comprehensive analysis of the theoretical capabilities of GNNs for various datasets, tasks, and scores, using an extension of the Weisfeiler-Lehman framework.

If you use Graphtester in your research, please cite the following paper:

```bibtex
@inproceedings{akbiyik2023graphtester,
    title        = {{Graphtester: Exploring Theoretical Boundaries of GNNs on Graph Datasets}},
    author       = {Eren Akbiyik and Florian Grötschla and Béni Egressy and Roger Wattenhofer},
    year         = 2023,
    month        = {July},
    booktitle    = {{Data-centric Machine Learning Research (DMLR) Workshop at ICML 2023, Honolulu, Hawaii}}
}
```

See documentation at [https://graphtester.readthedocs.io](https://graphtester.readthedocs.io).

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

# graphtester

## Installation

Simply run

```bash
pip install git+https://github.com/meakbiyik/graphtester.git
```

## Reproducing the results in the paper

1. Install the package
2. Run `sbatch scripts/paper/test_all_datasets.sh` to run all experiments on the datasets. Note that this scripts requires 10 cores and susntabtial amount of memory. You can change the number of cores by changing the `--cpus-per-task` argument in the script.
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

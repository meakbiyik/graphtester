# graphtester

## Contribution

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

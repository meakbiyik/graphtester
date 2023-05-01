#!/bin/bash
#SBATCH --job-name=test_all_datasets
#SBATCH --output=scripts/logs/paper/test_all_datasets.out
#SBATCH --error=scripts/logs/paper/test_all_datasets.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12

# Run the test_all_datasets.py script
poetry run python ./scripts/paper/test_all_datasets.py

#!/bin/bash
#SBATCH --job-name=graphtester
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12

# Run the test_all_datasets.py script
poetry run python ./scripts/paper/test_all_datasets.py

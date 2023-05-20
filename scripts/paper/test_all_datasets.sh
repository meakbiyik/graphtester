#!/bin/bash
#SBATCH --job-name=graphtester
#SBATCH --mem=250G
#SBATCH --cpus-per-task=8

# Run the test_all_datasets.py script
yes | poetry run python ./scripts/paper/test_all_datasets.py


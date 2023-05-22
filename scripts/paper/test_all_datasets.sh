#!/bin/bash
#SBATCH --job-name=graphtester
#SBATCH --mem=250G
#SBATCH --cpus-per-task=10

# Run the test_all_datasets.py script
source $(poetry env info --path)/bin/activate
yes | python ./scripts/paper/test_all_datasets.py

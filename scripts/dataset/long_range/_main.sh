#!/bin/bash
#SBATCH --job-name=DATA_LR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

# Load module and environment
module load python/3.8
source $PWD/tgrab/bin/activate
cd ../

python -m TSA.dataset.DTDG.graph_generation.run long_range \
    --num-nodes=$2 \
    --dataset-name="${1}" \
    --seed=12345 \
    \
    --val-ratio=$3 \
    --test-ratio=$4 \
    \
    --num-branches=$5 \
    --num-samples=$6 \
    --save-dir=$PWD/T-GRAB/data/
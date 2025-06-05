#!/bin/bash
#SBATCH --job-name=DATA_LR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

# Load module and environment
module load python/3.8
source $PWD/tgrab/bin/activate
cd ../

LAG=1
BRANCH_LEN=4

python -m T-GRAB.dataset.DTDG.graph_generation.run long_range \
    --num-nodes=100 \
    --dataset-name="($LAG, $BRANCH_LEN)" \
    --seed=12345 \
    \
    --val-ratio=0.1 \
    --test-ratio=0.1 \
    \
    --num-branches=3 \
    --num-samples=4000 \
    --visualize \
    --save-dir=$PWD/T-GRAB/scratch/data/
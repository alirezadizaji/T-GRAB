#!/bin/bash
#SBATCH --job-name=DATA_FIXED_ER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

# Load module and environment
module load python/3.8
source $PWD/tgrab/bin/activate
cd ../

python -m T-GRAB.dataset.DTDG.graph_generation.run periodicity \
    --num-nodes=$1 \
    --dataset-name="$7" \
    --seed=12345 \
    --num-of-training-weeks=$3 \
    --num-of-valid-weeks=$4 \
    --num-of-test-weeks=$5 \
    \
    --fixed-er-prob=$6 \
    --topology-mode=$2 \
    --visualize \
    --save-dir=$PWD/T-GRAB/data/


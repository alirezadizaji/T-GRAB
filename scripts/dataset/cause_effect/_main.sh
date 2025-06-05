#!/bin/bash
#SBATCH --job-name=DATA_CE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

# Load module and environment
source $PWD/tgrab/bin/activate
cd ../

python -m T-GRAB.dataset.DTDG.graph_generation.run cause_effect \
    --num-nodes=$2 \
    --dataset-name="${1}" \
    --seed=12345 \
    --val-ratio=$3 \
    --test-ratio=$4 \
    --test-inductive-ratio=$5 \
    --test-inductive-num-nodes-ratio=$6 \
    \
    --er-prob=$7 \
    --er-prob-inductive=$8 \
    --save-dir=$PWD/T-GRAB/scratch/data/

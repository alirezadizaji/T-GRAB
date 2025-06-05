#!/bin/bash
#SBATCH --job-name=DATA_LR_ER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

RUN_SCRIPT=long_range

# Load module and environment
module load python/3.8
source $HOME/envs/tsa/bin/activate
cd $HOME/lab

python -m TSA.dataset.DTDG.graph_generation.run $RUN_SCRIPT \
    --num-nodes=$2 \
    --dataset-name="${1}" \
    --neg-sampling-strategy="rnd" \
    --seed=12345 \
    --pattern-mode=$3 \
    --val-ratio=$4 \
    --test-ratio=$5 \
    \
    --er-prob-pattern=$6 \
    --er-pattern-extra=$7 \
    --visualize


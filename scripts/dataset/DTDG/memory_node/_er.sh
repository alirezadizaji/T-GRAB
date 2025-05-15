#!/bin/bash
#SBATCH --job-name=DATA_MN_ER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

RUN_SCRIPT=mem_node

# Load module and environment
module load python/3.8
source $HOME/envs/tsa/bin/activate
cd $HOME/lab

python -m TSA.dataset.DTDG.graph_generation.run $RUN_SCRIPT \
    --num-nodes=$2 \
    --dataset-name="${1}" \
    --neg-sampling-strategy="rnd" \
    --t-unit=$9 \
    --seed=12345 \
    --pattern-mode="er" \
    --val-ratio=$3 \
    --test-ratio=$4 \
    --test-inductive-ratio=$5 \
    --test-inductive-num-nodes-ratio=$6 \
    \
    --er-prob=$7 \
    --er-prob-inductive=$8

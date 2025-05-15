#!/bin/bash
#SBATCH --job-name=DATA_LR_SV2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

RUN_SCRIPT=spatial_longrange_cause_effect_v2

# Load module and environment
module load python/3.8
source $HOME/envs/tsa/bin/activate
cd $HOME/lab

python -m TSA.dataset.DTDG.graph_generation.run $RUN_SCRIPT \
    --num-nodes=$2 \
    --dataset-name="${1}" \
    --neg-sampling-strategy="rnd" \
    --seed=12345 \
    \
    --val-ratio=$3 \
    --test-ratio=$4 \
    \
    --num-branches=$5 \
    --num-active-branches=$6 \
    --num-samples=$7
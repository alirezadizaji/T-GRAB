#!/bin/bash
#SBATCH --job-name=DATA_FIXED_ER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

RUN_SCRIPT=periodicity

# Load module and environment
module load python/3.8
source $HOME/envs/tsa/bin/activate
cd $HOME/lab

python -m TSA.dataset.DTDG.graph_generation.run $RUN_SCRIPT \
    --num-nodes=$1 \
    --dataset-name="${8}" \
    --neg-sampling-strategy="rnd" \
    --seed=12345 \
    --num-of-training-weeks=$4 \
    --num-of-valid-weeks=$5 \
    --num-of-test-weeks=$6 \
    \
    --probability=0 \
    --fixed-er-prob=${7} \
    --topology-mode=$2 \
    --pruning-mode=$3 \
    --visualize


#!/bin/bash
#SBATCH --job-name=DATA_SBM_STO
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

RUN_SCRIPT=periodicity
# NUM_CLUSTERS=${8}
# NUM_PATTERNS=${14}

# Load module and environment
module load python/3.8
source $HOME/envs/tsa/bin/activate
cd $HOME/lab

# Read in script parameters
NUM_NODES=$1
NUM_CLUSTERS=${2}
PRUNING_MODE=$4
TOPOLOGY_MODE="$3"
NUM_TRAINING_WEEKS=$5
NUM_VALID_WEEKS=$6
NUM_TEST_WEEKS=$7
INTER_CLUSTER_PROB=${8}
dataset_pattern_indices="${9}"
INTRA_CLUSTER_PROB=${10}
# ADDITIONAL_INTRA_CLUSTER_PROB=${13}
# ADDITIONAL_ER_PROB=${13}

python -m TSA.dataset.DTDG.graph_generation.run $RUN_SCRIPT \
    --num-nodes=$NUM_NODES \
    --dataset-name="$dataset_pattern_indices" \
    --neg-sampling-strategy="rnd" \
    --seed=12345 \
    --num-of-training-weeks=$NUM_TRAINING_WEEKS \
    --num-of-valid-weeks=$NUM_VALID_WEEKS \
    --num-of-test-weeks=$NUM_TEST_WEEKS \
    \
    --visualize \
    --topology-mode="$TOPOLOGY_MODE" \
    --pruning-mode=$PRUNING_MODE \
    --num-clusters ${NUM_CLUSTERS//_/ } \
    --intra-cluster-prob=$INTRA_CLUSTER_PROB \
    --inter-cluster-prob=$INTER_CLUSTER_PROB \
    --probability=0
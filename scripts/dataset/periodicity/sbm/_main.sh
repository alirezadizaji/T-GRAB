#!/bin/bash
#SBATCH --job-name=DATA_SBM
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

# Load module and environment
module load python/3.8
source $PWD/tgrab/bin/activate
cd ../

# Read in script parameters
NUM_NODES=$1
NUM_CLUSTERS=${2}
TOPOLOGY_MODE="$3"
NUM_TRAINING_WEEKS=$4
NUM_VALID_WEEKS=$5
NUM_TEST_WEEKS=$6
INTER_CLUSTER_PROB=$7
dataset_pattern_indices="$8"
INTRA_CLUSTER_PROB=$9

python -m T-GRAB.dataset.DTDG.graph_generation.run periodicity \
    --num-nodes=$NUM_NODES \
    --dataset-name="$dataset_pattern_indices" \
    --seed=12345 \
    --num-of-training-weeks=$NUM_TRAINING_WEEKS \
    --num-of-valid-weeks=$NUM_VALID_WEEKS \
    --num-of-test-weeks=$NUM_TEST_WEEKS \
    \
    --visualize \
    --topology-mode="$TOPOLOGY_MODE" \
    --num-clusters=$NUM_CLUSTERS \
    --intra-cluster-prob=$INTRA_CLUSTER_PROB \
    --inter-cluster-prob=$INTER_CLUSTER_PROB \
    --save-dir=$PWD/T-GRAB/scratch/data/
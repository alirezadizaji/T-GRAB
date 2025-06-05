#!/bin/bash

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

module load python/3.8
source $PWD/tgrab/bin/activate
cd ../

K=2
N=1

python -m T-GRAB.dataset.DTDG.graph_generation.run periodicity \
    --num-nodes=100 \
    --dataset-name="($K, $N)" \
    --seed=12345 \
    --num-of-training-weeks=40 \
    --num-of-valid-weeks=4 \
    --num-of-test-weeks=4 \
    \
    --visualize \
    --topology-mode="sbm" \
    --num-clusters=3 \
    --intra-cluster-prob=0.9 \
    --inter-cluster-prob=0.01 \
    --save-dir=$PWD/T-GRAB/data/
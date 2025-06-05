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
    --fixed-er-prob=0.01 \
    --topology-mode="fixed_er" \
    --visualize \
    --save-dir=$PWD/T-GRAB/data/


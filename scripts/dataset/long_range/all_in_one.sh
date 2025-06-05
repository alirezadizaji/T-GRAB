#!/bin/bash

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

echo "Start submitting long-range dataset generation..."
sleep 2

NUM_BRANCHES=3
NUM_NODES=100

for NUM_SAMPLES in 4000
do
    for VAL_RATIO in 0.1
    do
        for TEST_RATIO in 0.1
        do
            for LAG in 1 4 16
            do
                for BRANCH_LEN in 1 4 16
                do
                    DATASET_PATTERN="($LAG, $BRANCH_LEN)"
                    sbatch \
                        --mem=4G \
                        --output="logs/$PATTERN_MODE/($LAG, $BRANCH_LEN)_${NUM_NODES}nn_${NUM_BRANCHES}nb_${NUM_SAMPLES}ns/%j-e.out" \
                        --error="logs/$PATTERN_MODE/($LAG, $BRANCH_LEN)_${NUM_NODES}nn_${NUM_BRANCHES}nb_${NUM_SAMPLES}ns/%j-o.out" \
                        _main.sh \
                            "$DATASET_PATTERN" \
                            $NUM_NODES \
                            $VAL_RATIO \
                            $TEST_RATIO \
                            \
                            $NUM_BRANCHES \
                            $NUM_SAMPLES
                done
            done
        done
    done
done


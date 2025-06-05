#!/bin/bash

cd $HOME/lab/TSA/scripts/dataset/DTDG/long_range
which_dataset_to_generate=("$@")

for PATTERN_MODE in "${which_dataset_to_generate[@]}"; do

    ## SCEU V3
    if [[ "$PATTERN_MODE" == "SCEU_V3" ]]; then
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
                                _slrce_v3.sh \
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
    fi
done


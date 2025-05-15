#!/bin/bash

cd $HOME/lab/TSA/scripts/dataset/DTDG/long_range
which_dataset_to_generate=("$@")

for PATTERN_MODE in "${which_dataset_to_generate[@]}"; do

    ## fixed er
    if [[ "$PATTERN_MODE" == "er" ]]; then
        echo "Start submitting er long range generation..."
        sleep 2
        NUM_NODES=100

        for VAL_RATIO in 0.1
        do
            for TEST_RATIO in 0.1
            do
                for ER_PROB_PATTERN in 0.4
                do
                    for ER_PROB_EXTRA in 0.01
                    do
                        for FREQ_LEN in 16 32 64 128 256 512
                        do
                            for REPETITION in 100 200 400
                            do
                                DATASET_PATTERN="($FREQ_LEN, $REPETITION)"
                                sbatch \
                                    --mem=4G \
                                    --output="logs/er/($FREQ_LEN, $REPETITION)_${ER_PROB_PATTERN}_${ER_PROB_EXTRA}/%j-e.out" \
                                    --error="logs/er/($FREQ_LEN, $REPETITION)_${ER_PROB_PATTERN}_${ER_PROB_EXTRA}/%j-o.out" \
                                    _er.sh \
                                        "$DATASET_PATTERN" \
                                        $NUM_NODES \
                                        $PATTERN_MODE \
                                        $VAL_RATIO \
                                        $TEST_RATIO \
                                        \
                                        $ER_PROB_PATTERN \
                                        $ER_PROB_EXTRA
                            done
                        done
                    done
                done
            done
        done
    fi

    ## fixed er
    if [[ "$PATTERN_MODE" == "SCEU" ]]; then
        PATTERN_NUM_NODES=100

        for NUM_SAMPLES in 4000
        do
            for VAL_RATIO in 0.1
            do
                for TEST_RATIO in 0.1
                do
                    for TEST_INDUCTIVE_RATIO in 0.1
                    do
                        for TEST_INDUCTIVE_NUM_NODES_RATIO in 0.1
                        do
                            for NUM_EDGES in 20
                            do
                                for NUM_EDGES_INDUCTIVE in 2
                                do
                                    for LAG in 1
                                    do
                                        for DIST in 255
                                        do
                                            NUM_NODES=$((PATTERN_NUM_NODES + DIST + 1))
                                            DATASET_PATTERN="($LAG, $DIST)"
                                            # sbatch \
                                            #     --mem=4G \
                                            #     --output="logs/$PATTERN_MODE/($LAG, $DIST)_${NUM_EDGES}ne_${NUM_EDGES_INDUCTIVE}nei/%j-e.out" \
                                            #     --error="logs/$PATTERN_MODE/($LAG, $DIST)_${NUM_EDGES}ne_${NUM_EDGES_INDUCTIVE}nei/%j-o.out" \
                                                ./_spatial_cause_effect_uniform.sh \
                                                    "$DATASET_PATTERN" \
                                                    $NUM_NODES \
                                                    $VAL_RATIO \
                                                    $TEST_RATIO \
                                                    $TEST_INDUCTIVE_RATIO \
                                                    $TEST_INDUCTIVE_NUM_NODES_RATIO \
                                                    \
                                                    $NUM_EDGES \
                                                    $NUM_EDGES_INDUCTIVE \
                                                    $NUM_SAMPLES
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    fi
    
    ## fixed er
    if [[ "$PATTERN_MODE" == "SCEU_V2" ]]; then
        NUM_BRANCHES=10

        for NUM_SAMPLES in 4000
        do
            for VAL_RATIO in 0.1
            do
                for TEST_RATIO in 0.1
                do
                    for NUM_ACTIVE_BRANCHES in 2
                    do
                        for LAG in 5
                        do
                            for BRANCH_LEN in 2 4
                            do
                                DATASET_PATTERN="($LAG, $BRANCH_LEN)"
                                NUM_NODES=$((NUM_BRANCHES * BRANCH_LEN + 2))
                                echo $NUM_NODES
                                sbatch \
                                    --mem=4G \
                                    --output="logs/$PATTERN_MODE/($LAG, $BRANCH_LEN)_${NUM_ACTIVE_BRANCHES}anb_${NUM_BRANCHES}nb_${NUM_SAMPLES}ns/%j-e.out" \
                                    --error="logs/$PATTERN_MODE/($LAG, $BRANCH_LEN)_${NUM_ACTIVE_BRANCHES}anb_${NUM_BRANCHES}nb_${NUM_SAMPLES}ns/%j-o.out" \
                                    _slrce_v2.sh \
                                        "$DATASET_PATTERN" \
                                        $NUM_NODES \
                                        $VAL_RATIO \
                                        $TEST_RATIO \
                                        \
                                        $NUM_BRANCHES \
                                        $NUM_ACTIVE_BRANCHES \
                                        $NUM_SAMPLES
                            done
                        done
                    done
                done
            done
        done
    fi
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


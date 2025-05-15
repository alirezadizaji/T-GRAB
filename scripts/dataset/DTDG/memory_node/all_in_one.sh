#!/bin/bash

cd $HOME/lab/TSA/scripts/dataset/DTDG/memory_node
which_dataset_to_generate=("$@")

for PATTERN_MODE in "${which_dataset_to_generate[@]}"; do
    echo "Start submitting $PATTERN_MODE memory node generation..."
    sleep 2

    ## fixed er
    if [[ "$PATTERN_MODE" == "er" ]]; then
        NUM_NODES=101

        for VAL_RATIO in 0.1
        do
            for TEST_RATIO in 0.1
            do
                for TEST_INDUCTIVE_RATIO in 0.1
                do
                    for TEST_INDUCTIVE_NUM_NODES_RATIO in 0.1
                    do
                        for ER_PROB in 0.002
                        do
                            for ER_PROB_INDUCTIVE in 0.02
                            do
                                for GAP in 4
                                do
                                    for NUM_PATTERNS in 4000
                                    do
                                        DATASET_PATTERN="($GAP, $NUM_PATTERNS)"
                                        T_UNIT=4
                                        # sbatch \
                                        #     --mem=4G \
                                        #     --output="logs/$PATTERN_MODE/($GAP, $NUM_PATTERNS)_${ER_PROB}_${ER_PROB_INDUCTIVE}/%j-e.out" \
                                        #     --error="logs/$PATTERN_MODE/($GAP, $NUM_PATTERNS)_${ER_PROB}_${ER_PROB_INDUCTIVE}/%j-o.out" \
                                            ./_er.sh \
                                                "$DATASET_PATTERN" \
                                                $NUM_NODES \
                                                $VAL_RATIO \
                                                $TEST_RATIO \
                                                $TEST_INDUCTIVE_RATIO \
                                                $TEST_INDUCTIVE_NUM_NODES_RATIO \
                                                \
                                                $ER_PROB \
                                                $ER_PROB_INDUCTIVE \
                                                $T_UNIT
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    fi
done


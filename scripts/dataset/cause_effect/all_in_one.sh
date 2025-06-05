#!/bin/bash

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

echo "Start submitting cause-and-effect dataset generation..."
sleep 2

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
                        for LAG in 4
                        do
                            for NUM_PATTERNS in 4000
                            do
                                DATASET_PATTERN="($LAG, $NUM_PATTERNS)"
                                # sbatch \
                                #     --mem=4G \
                                #     --output="logs/$PATTERN_MODE/($LAG, $NUM_PATTERNS)_${ER_PROB}_${ER_PROB_INDUCTIVE}/%j-e.out" \
                                #     --error="logs/$PATTERN_MODE/($LAG, $NUM_PATTERNS)_${ER_PROB}_${ER_PROB_INDUCTIVE}/%j-o.out" \
                                    scripts/dataset/cause_effect/_main.sh \
                                        "$DATASET_PATTERN" \
                                        $NUM_NODES \
                                        $VAL_RATIO \
                                        $TEST_RATIO \
                                        $TEST_INDUCTIVE_RATIO \
                                        $TEST_INDUCTIVE_NUM_NODES_RATIO \
                                        \
                                        $ER_PROB \
                                        $ER_PROB_INDUCTIVE
                            done
                        done
                    done
                done
            done
        done
    done
done


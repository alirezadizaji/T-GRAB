#!/bin/bash

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

which_dataset_to_generate=("$@")

for value in "${which_dataset_to_generate[@]}"; do

    ## fixed er
    if [[ "$value" == "fixed_er" ]]; then
        echo "Start submitting fixed_er periodicity generation..."
        sleep 2
        NUM_NODES=100
        TOPOLOGY_MODE=$value

        for PROB in 0.01
        do
            for NUM_TRAINING_WEEKS in 40
            do
                for NUM_VALID_WEEKS in 4
                do
                    for NUM_TEST_WEEKS in 4
                    do
                        for K in 4
                        do
                            for N in 1
                            do
                                DATASET_PATTERN="($K, $N)"
                                sbatch \
                                    --mem=4G \
                                    --output="logs/fixed_er/($K, $N)_$PROB/%j-e.out" \
                                    --error="logs/fixed_er/($K, $N)_$PROB/%j-o.out" \
                                    scripts/dataset/periodicity/fixed_er/_main.sh \
                                        $NUM_NODES \
                                        $TOPOLOGY_MODE \
                                        $NUM_TRAINING_WEEKS \
                                        $NUM_VALID_WEEKS \
                                        $NUM_TEST_WEEKS \
                                        $PROB \
                                        "$DATASET_PATTERN"
                            done
                        done
                    done
                done
            done
        done
    fi

    if [[ "$value" == "sbm" ]]; then
        echo "Start submitting sbm stochastic periodicity generation..."
        sleep 2
        NUM_NODES=100
        TOPOLOGY_MODE=$value

        for INTER_CLUSTER_PROB in 0.0
        do
            for NUM_TRAINING_WEEKS in 40 
            do
                for NUM_VALID_WEEKS in 4
                do
                    for NUM_TEST_WEEKS in 4
                    do
                        for NUM_CLUSTERS in "3"
                        do
                            for INTRA_CLUSTER_PROB in 1.0
                            do
                                for K in 3
                                do
                                    for N in 1
                                    do
                                        PERIODIC_PATTERN="($K, $N)"
                                        # sbatch \
                                        #     --mem=4G \
                                        #     --output="logs/sbm_sto/($K, $N)/${NUM_TRAINING_WEEKS}trW-${NUM_CLUSTERS}nc-${INTER_CLUSTER_PROB}icnp-${INTRA_CLUSTER_PROB}icp/%j-e.out" \
                                        #     --error="logs/sbm_sto/($K, $N)/${NUM_TRAINING_WEEKS}trW-${NUM_CLUSTERS}nc-${INTER_CLUSTER_PROB}icnp-${INTRA_CLUSTER_PROB}icp/%j-o.out" \
                                            ./scripts/dataset/periodicity/sbm/_main.sh $NUM_NODES \
                                                $NUM_CLUSTERS $TOPOLOGY_MODE $NUM_TRAINING_WEEKS $NUM_VALID_WEEKS $NUM_TEST_WEEKS \
                                                $INTER_CLUSTER_PROB "$PERIODIC_PATTERN" $INTRA_CLUSTER_PROB
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


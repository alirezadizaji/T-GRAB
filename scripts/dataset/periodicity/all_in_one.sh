#!/bin/bash

cd $HOME/lab/TSA/scripts/dataset/DTDG/periodicity
which_dataset_to_generate=("$@")

for value in "${which_dataset_to_generate[@]}"; do

    ## fixed er
    if [[ "$value" == "fixed_er" ]]; then
        echo "Start submitting fixed_er periodicity generation..."
        sleep 2
        NUM_NODES=100
        TOPOLOGY_MODE="fixed_er"
        PRUNING_MODE="none"

        for PROB in 0.01
        do
            for NUM_TRAINING_WEEKS in 40
            do
                for NUM_VALID_WEEKS in 4
                do
                    for NUM_TEST_WEEKS in 4
                    do
                        for K in 4 8 16 32 64 128
                        do
                            for N in 4
                            do
                                DATASET_PATTERN="($K, $N)"
                                sbatch \
                                    --mem=4G \
                                    --output="logs/fixed_er/($K, $N)_$PROB/%j-e.out" \
                                    --error="logs/fixed_er/($K, $N)_$PROB/%j-o.out" \
                                    _fixed_er.sh \
                                        $NUM_NODES \
                                        $TOPOLOGY_MODE \
                                        $PRUNING_MODE \
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

    if [[ "$value" == "sbm_sto" ]]; then
        echo "Start submitting sbm stochastic periodicity generation..."
        sleep 2
        NUM_NODES=100
        TOPOLOGY_MODE="sbm_sto"
        PRUNING_MODE="none"

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
                                for K in 2 4 8 16 32 64 128 256
                                do
                                    for N in 1
                                    do
                                        PERIODIC_PATTERN="($K, $N)"
                                        sbatch \
                                            --mem=4G \
                                            --output="logs/sbm_sto/($K, $N)/${NUM_TRAINING_WEEKS}trW-${NUM_CLUSTERS}nc-${INTER_CLUSTER_PROB}icnp-${INTRA_CLUSTER_PROB}icp/%j-e.out" \
                                            --error="logs/sbm_sto/($K, $N)/${NUM_TRAINING_WEEKS}trW-${NUM_CLUSTERS}nc-${INTER_CLUSTER_PROB}icnp-${INTRA_CLUSTER_PROB}icp/%j-o.out" \
                                            _sbm_sto.sh $NUM_NODES \
                                                $NUM_CLUSTERS $TOPOLOGY_MODE $PRUNING_MODE $NUM_TRAINING_WEEKS $NUM_VALID_WEEKS $NUM_TEST_WEEKS \
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


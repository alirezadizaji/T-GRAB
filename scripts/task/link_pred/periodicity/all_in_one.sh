export PROJ_ROOT_DIR="$HOME/lab"
export ROOT_LOAD_SAVE_DIR="$SCRATCH/"
export SCRIPT_LOC=scripts/task/link_pred/
export DATA_LOC=lab/TSA/data/
export PYENV=$HOME/envs/tsa/



cd $PROJ_ROOT_DIR/TSA/$SCRIPT_LOC

task=periodicity
NUM_EPOCHS_TO_VIS=0
which_dataset_to_train=("$@")
DYGLIB=("CTDG/_dygformer" "CTDG/_tgn" "CTDG/_tgat")

# User can change these variables.
EVAL_MODE=false
CTDG_DO_SNAPSHOT_TRAINING=true
METHODS_TO_RUN=("CTDG/_dygformer")
CLEAR_RESULT=false 
WANDB_ENTITY="alirezadizaji24-universit-de-montr-al"

for value in "${which_dataset_to_train[@]}"; 
do
    ## Fixed_er training
    if [[ "$value" == "fixed_er" ]]; then
        VAL_FIRST_METRIC="avg_f1"
        EVAL_WEEK=4
        #Periodicity training        
        for FIXED_PROB in 0.01
        do
            for NUM_TRAINING_WEEKS in 40
            do
                for K in 256
                do
                    for N in 1
                    do
                        let PERIOD_LEN=$((K * N))
                        # Initialize an empty string
                        DATASET_PATTERN="($K, $N)"

                        DATA="$DATASET_PATTERN/fixed_er-100n-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-p0.0-fp${FIXED_PROB}"
                
                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                            # Edgebank doesn't need seed, or node_feat
                            sbatch \
                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${edgebank_model}/slurm-${NUM_TRAINING_WEEKS}trW-${NUM_REPS}r-${FIXED_PROB}fp-%j-o.out" \
                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${edgebank_model}/slurm-${NUM_TRAINING_WEEKS}trW-${NUM_REPS}r-${FIXED_PROB}fp-%j-e.out" \
                                $task/CTDG/_edgebank.sh \
                                    "$DATA" \
                                    $ROOT_LOAD_SAVE_DIR \
                                    "$VAL_FIRST_METRIC"
                        fi

                        # Compute memory
                        # Following formula was found empirically to avoid oom in all cases.
                        RAW_MEM=$(echo "0.32 * $NUM_TRAINING_WEEKS * $PERIOD_LEN * $FIXED_PROB" | bc)
                        RAW_MEM=$(printf "%.0f" "$RAW_MEM")

                        for SEED in 3457
                        do
                            for NODE_FEAT in "ONE_HOT"
                            do
                                # As far as NODE_FEAT=ONE_HOT, it's not important what is the node feature dimension!
                                for NODE_FEAT_DIM in 1
                                do
                                    # Continuous-time dynamic graph methods implemented by DyGLib.
                                    for model in "CTDG/_tgat" "CTDG/_tgn" "CTDG/_dygformer" "CTDG/_tgn_tgb"
                                    do
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                            # Memory computation for methods implemented by DyGLib
                                            if (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                                MEM=4
                                            else
                                                MEM=$RAW_MEM
                                            fi
                                            MAX_GPU=80

                                            # Compute required GPU
                                            # If required memory is more than 160G, then assign maximum available GPU(80G)
                                            GPU=$((MAX_GPU * MEM / 160))
                                            # Apply conditions
                                            if (($GPU > 64)); then
                                                let GPU=80
                                            elif (($GPU > 44)); then
                                                let GPU=48
                                            elif (($GPU > 36)); then
                                                let GPU=40
                                            else
                                                let GPU=32
                                            fi

                                            GPU=$(printf "%.0f" "$GPU")
                                            for NUM_UNITS in 1
                                            do
                                                for NUM_HEADS in 2
                                                do
                                                    for TIME_FEAT_DIM in 100
                                                    do
                                                        for NUM_NEIGHBORS in 20
                                                        do
                                                            for CHANNEL_EMBEDDING_DIM in 50
                                                            do
                                                                for MAX_INPUT_SEQ_LEN in 32 64 128 256 512
                                                                do
                                                                    for TRAIN_BATCH_SIZE in 1
                                                                    do
                                                                        for MEMORY_DIM in 100
                                                                        do
                                                                            sbatch \
                                                                                --mem=${MEM}gb \
                                                                                --gres=gpu:${GPU}gb:1 \
                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-${FIXED_PROB}p-%j-o.out" \
                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-${FIXED_PROB}p-%j-e.out" \
                                                                                $task/$model.sh \
                                                                                    "$DATA" \
                                                                                    $SEED \
                                                                                    $NODE_FEAT \
                                                                                    $NODE_FEAT_DIM \
                                                                                    $EVAL_MODE \
                                                                                    $NUM_EPOCHS_TO_VIS \
                                                                                    $ROOT_LOAD_SAVE_DIR \
                                                                                    "$VAL_FIRST_METRIC" \
                                                                                    $MEM \
                                                                                    $MAX_GPU \
                                                                                    $GPU \
                                                                                    $NUM_UNITS \
                                                                                    $NUM_HEADS \
                                                                                    $TIME_FEAT_DIM \
                                                                                    $NUM_NEIGHBORS \
                                                                                    $TRAIN_BATCH_SIZE \
                                                                                    $CTDG_DO_SNAPSHOT_TRAINING \
                                                                                    $CHANNEL_EMBEDDING_DIM \
                                                                                    $MAX_INPUT_SEQ_LEN \
                                                                                    $MEMORY_DIM \
                                                                                    $CLEAR_RESULT \
                                                                                    $WANDB_ENTITY
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

                                    # CTDG/_ctan
                                    if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_ctan " ]]; then
                                        # Memory computation for CTAN
                                        if (( $(echo "$RAW_MEM > 40" | bc -l) )); then
                                            MEM=40
                                        elif (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                            MEM=4
                                        else
                                            MEM=$RAW_MEM
                                        fi
                                        MAX_GPU=48

                                        # Compute required GPU
                                        # If required memory is more than 160G, then assign maximum available GPU(80G)
                                        GPU=$((MAX_GPU * MEM / 160))
                                        # Apply conditions
                                        if (($GPU > 44)); then
                                            let GPU=48
                                        elif (($GPU > 36)); then
                                            let GPU=40
                                        else
                                            let GPU=32
                                        fi

                                        GPU=$(printf "%.0f" "$GPU")
                                        
                                        for NUM_UNITS in 1 
                                        do
                                            for OUT_CHANNELS in 128
                                            do
                                                for TIME_FEAT_DIM in 100
                                                do
                                                    for SAMPLER_SIZE in 8
                                                    do
                                                        for TRAIN_BATCH_SIZE in 1
                                                        do
                                                            sbatch \
                                                                --mem=${MEM}gb \
                                                                --gres=gpu:${GPU}gb:1 \
                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/CTDG/_ctan/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-${FIXED_PROB}p-%j-o.out" \
                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/CTDG/_ctan/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-${FIXED_PROB}p-%j-e.out" \
                                                                $task/CTDG/_ctan.sh \
                                                                    "$DATA" \
                                                                    $SEED \
                                                                    $NODE_FEAT \
                                                                    $NODE_FEAT_DIM \
                                                                    $EVAL_MODE \
                                                                    $NUM_EPOCHS_TO_VIS \
                                                                    $ROOT_LOAD_SAVE_DIR \
                                                                    "$VAL_FIRST_METRIC" \
                                                                    $MEM \
                                                                    $MAX_GPU \
                                                                    $GPU \
                                                                    $NUM_UNITS \
                                                                    $OUT_CHANNELS \
                                                                    $TIME_FEAT_DIM \
                                                                    $SAMPLER_SIZE \
                                                                    $TRAIN_BATCH_SIZE \
                                                                    $CTDG_DO_SNAPSHOT_TRAINING \
                                                                    $CLEAR_RESULT \
                                                                    $WANDB_ENTITY
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    fi

                                    # Discrete-time dynamic graph methods
                                    for model in "DTDG/_gcn" "DTDG/_gclstm" "DTDG/_egcno" "DTDG/_tgcn" "DTDG/_gat" "DTDG/_egcnh"
                                    do
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                            for NUM_UNITS in 1
                                            do
                                                for OUT_CHANNELS in 128
                                                do
                                                    if (( $(echo "$RAW_MEM > 16" | bc -l) )); then
                                                        MEM=16
                                                    elif (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                                        MEM=4
                                                    else
                                                        MEM=$RAW_MEM
                                                    fi

                                                    sbatch \
                                                        --mem=${MEM}gb \
                                                        --gres=gpu:32gb:1 \
                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-${FIXED_PROB}p-%j-o.out" \
                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-${FIXED_PROB}p-%j-e.out" \
                                                        $task/$model.sh \
                                                            "$DATA" \
                                                            $SEED \
                                                            $NODE_FEAT \
                                                            $NODE_FEAT_DIM \
                                                            $EVAL_MODE \
                                                            $NUM_EPOCHS_TO_VIS \
                                                            $ROOT_LOAD_SAVE_DIR \
                                                            "$VAL_FIRST_METRIC" \
                                                            $OUT_CHANNELS \
                                                            $NUM_UNITS \
                                                            $CLEAR_RESULT \
                                                            $WANDB_ENTITY
                                                done
                                            done
                                        fi
                                    done
            
                                    # Baseline models
                                    for model in "DTDG/_empty" "DTDG/_clique" "DTDG/_previous"
                                    do
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                            sbatch \
                                                --mem=4g \
                                                --partition=long-cpu \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${model}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-%j-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${model}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-%j-e.out" \
                                                $task/$model.sh \
                                                    "$DATA" \
                                                    $ROOT_LOAD_SAVE_DIR \
                                                    $WANDB_ENTITY
                                        fi
                                    done

                                done
                            done
                        done
                    done
                done
            done
        done
    fi

    ## sbm stochastic training
    if [[ "$value" == "sbm_sto" ]]; then
        VAL_FIRST_METRIC="avg_f1"
        NUM_NODES=100
        #Periodicity training
        for NUM_TRAINING_WEEKS in 40
        do
            for NUM_VALID_WEEKS in 4
            do
                for NUM_TEST_WEEKS in 4
                do
                    for INTER_CLUSTER_PROB in 0.01
                    do
                        for NUM_CLUSTERS in "2_10"
                        do
                            NUM_CLUSTERS=$(echo "$NUM_CLUSTERS" | sed 's/_/, /g' | sed 's/^/[/' | sed 's/$/]/')
                            for INTRA_CLUSTER_PROB in 0.1
                            do
                                for K in 2
                                do
                                    for N in 4 8
                                    do
                                        dataset_pattern_indices="($K, $N)"
                                        DATA="$dataset_pattern_indices/sbm_sto-${NUM_NODES}n-${NUM_TRAINING_WEEKS}trW-${NUM_VALID_WEEKS}vW-${NUM_TEST_WEEKS}tsW-p0.0-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}"
                                        
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                            # Edgebank doesn't need seed, or node_feat
                                            sbatch \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/CTDG/_edgebank/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/CTDG/_edgebank/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                $task/CTDG/_edgebank.sh \
                                                    "$DATA" \
                                                    $ROOT_LOAD_SAVE_DIR \
                                                    "$VAL_FIRST_METRIC"
                                        fi

                                        # Compute memory
                                        # Following formula was found empirically to avoid oom in all cases.
                                        RAW_MEM=$(echo "0.03 * $K * $N * $NUM_NODES * $NUM_TRAINING_WEEKS * $INTRA_CLUSTER_PROB * $INTER_CLUSTER_PROB" | bc)
                                        RAW_MEM=$(printf "%.0f" "$RAW_MEM")
                                        
                                        for SEED in 1235
                                        do
                                            for NODE_FEAT in "ONE_HOT"
                                            do
                                                # As far as NODE_FEAT=ONE_HOT, it's not important what is the node feature dimension!
                                                for NODE_FEAT_DIM in 1
                                                do
                                                    # DYGLIB training
                                                    for model in "CTDG/_dygformer" "CTDG/_tgn" "CTDG/_tgat" "CTDG/_tgn_tgb"
                                                    do
                                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                                            # Memory computation for methods implemented by DyGLib
                                                            if (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                                                MEM=4
                                                            else
                                                                MEM=$RAW_MEM
                                                            fi
                                                            GPU=40
                                                            MAX_GPU=40

                                                            for NUM_UNITS in 1
                                                            do
                                                                for NUM_HEADS in 2
                                                                do
                                                                    for TIME_FEAT_DIM in 100
                                                                    do
                                                                        for NUM_NEIGHBORS in 20
                                                                        do
                                                                            for CHANNEL_EMBEDDING_DIM in 50
                                                                            do
                                                                                for MAX_INPUT_SEQ_LEN in 20
                                                                                do
                                                                                    for TRAIN_BATCH_SIZE in 1
                                                                                    do
                                                                                        for MEMORY_DIM in 100
                                                                                        do
                                                                                            sbatch \
                                                                                                --mem=${MEM}gb \
                                                                                                --gres=gpu:1 \
                                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                                                                $task/$model.sh \
                                                                                                    "$DATA" \
                                                                                                    $SEED \
                                                                                                    $NODE_FEAT \
                                                                                                    $NODE_FEAT_DIM \
                                                                                                    $EVAL_MODE \
                                                                                                    $NUM_EPOCHS_TO_VIS \
                                                                                                    $ROOT_LOAD_SAVE_DIR \
                                                                                                    "$VAL_FIRST_METRIC" \
                                                                                                    $MEM \
                                                                                                    $MAX_GPU \
                                                                                                    $GPU \
                                                                                                    $NUM_UNITS \
                                                                                                    $NUM_HEADS \
                                                                                                    $TIME_FEAT_DIM \
                                                                                                    $NUM_NEIGHBORS \
                                                                                                    $TRAIN_BATCH_SIZE \
                                                                                                    $CTDG_DO_SNAPSHOT_TRAINING \
                                                                                                    $CHANNEL_EMBEDDING_DIM \
                                                                                                    $MAX_INPUT_SEQ_LEN \
                                                                                                    $MEMORY_DIM \
                                                                                                    $CLEAR_RESULT \
                                                                                                    $WANDB_ENTITY
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

                                                    # CTDG/_ctan
                                                    if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_ctan " ]]; then
                                                        # Memory computation for CTAN
                                                        if (( $(echo "$RAW_MEM > 40" | bc -l) )); then
                                                            MEM=40
                                                        elif (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                                            MEM=4
                                                        else
                                                            MEM=$RAW_MEM
                                                        fi
                                                        
                                                        GPU=40
                                                        MAX_GPU=40

                                                        for NUM_UNITS in 1 
                                                        do
                                                            for OUT_CHANNELS in 128
                                                            do
                                                                for TIME_FEAT_DIM in 100
                                                                do
                                                                    for SAMPLER_SIZE in 20
                                                                    do
                                                                        for TRAIN_BATCH_SIZE in 1
                                                                        do
                                                                            sbatch \
                                                                                --mem=${MEM}gb \
                                                                                --gres=gpu:1 \
                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/CTDG/_ctan/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/CTDG/_ctan/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                                                $task/CTDG/_ctan.sh \
                                                                                    "$DATA" \
                                                                                    $SEED \
                                                                                    $NODE_FEAT \
                                                                                    $NODE_FEAT_DIM \
                                                                                    $EVAL_MODE \
                                                                                    $NUM_EPOCHS_TO_VIS \
                                                                                    $ROOT_LOAD_SAVE_DIR \
                                                                                    "$VAL_FIRST_METRIC" \
                                                                                    $MEM \
                                                                                    $MAX_GPU \
                                                                                    $GPU \
                                                                                    $NUM_UNITS \
                                                                                    $OUT_CHANNELS \
                                                                                    $TIME_FEAT_DIM \
                                                                                    $SAMPLER_SIZE \
                                                                                    $TRAIN_BATCH_SIZE \
                                                                                    $CTDG_DO_SNAPSHOT_TRAINING \
                                                                                    $CLEAR_RESULT \
                                                                                    $WANDB_ENTITY
                                                                        done
                                                                    done
                                                                done
                                                            done
                                                        done
                                                    fi
                                                    
                                                    # Discrete-time dynamic graph methods
                                                    for model in "DTDG/_gcn" "DTDG/_gclstm" "DTDG/_egcno" "DTDG/_tgcn" "DTDG/_gat"
                                                    do
                                                        for NUM_UNITS in 1
                                                        do
                                                            for OUT_CHANNELS in 128
                                                            do
                                                                if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                                                    if (( $(echo "$RAW_MEM > 16" | bc -l) )); then
                                                                        MEM=16
                                                                    elif (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                                                        MEM=4
                                                                    else
                                                                        MEM=$RAW_MEM
                                                                    fi
                                                                    sbatch \
                                                                        --mem=${MEM}gb \
                                                                        --gres=gpu:1 \
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                                        $task/$model.sh \
                                                                            "$DATA" \
                                                                            $SEED \
                                                                            $NODE_FEAT \
                                                                            $NODE_FEAT_DIM \
                                                                            $EVAL_MODE \
                                                                            $NUM_EPOCHS_TO_VIS \
                                                                            $ROOT_LOAD_SAVE_DIR \
                                                                            "$VAL_FIRST_METRIC" \
                                                                            $OUT_CHANNELS \
                                                                            $NUM_UNITS \
                                                                            $CLEAR_RESULT \
                                                                            $WANDB_ENTITY
                                                                fi
                                                            done
                                                        done
                                                    done
                                                    # Baseline models
                                                    for model in "DTDG/_empty" "DTDG/_clique" "DTDG/_previous"  "CTDG/_sbm_bayes"
                                                    do
                                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                                            sbatch \
                                                                --mem=4gb \
                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                                $task/$model.sh \
                                                                "$DATA" \
                                                                $ROOT_LOAD_SAVE_DIR \
                                                                $WANDB_ENTITY
                                                        fi
                                                    done
                                                done
                                            done
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

    ## sbm stochastic training
    if [[ "$value" == "sbm_sto_v2" ]]; then
        VAL_FIRST_METRIC="avg_f1"
        NUM_NODES=100
        #Periodicity training
        for NUM_TRAINING_WEEKS in 40
        do
            for NUM_VALID_WEEKS in 4
            do
                for NUM_TEST_WEEKS in 4
                do
                    for INTER_CLUSTER_PROB in 0.01
                    do
                        for NUM_CLUSTERS in "3"
                        do
                            NUM_CLUSTERS=$(echo "$NUM_CLUSTERS" | sed 's/_/, /g' | sed 's/^/[/' | sed 's/$/]/')
                            for INTRA_CLUSTER_PROB in 0.9
                            do
                                for K in 8 
                                do
                                    for N in 1
                                    do
                                        dataset_pattern_indices="($K, $N)"
                                        DATA="$dataset_pattern_indices/sbm_sto_v2-${NUM_NODES}n-${NUM_TRAINING_WEEKS}trW-${NUM_VALID_WEEKS}vW-${NUM_TEST_WEEKS}tsW-p0.0-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}"
                                        
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                            # Edgebank doesn't need seed, or node_feat
                                            sbatch \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/CTDG/_edgebank/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/CTDG/_edgebank/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                $task/CTDG/_edgebank.sh \
                                                    "$DATA" \
                                                    $ROOT_LOAD_SAVE_DIR \
                                                    "$VAL_FIRST_METRIC"
                                        fi

                                        # Compute memory
                                        # Following formula was found empirically to avoid oom in all cases.
                                        RAW_MEM=$(echo "0.0006 * $K * $N * $NUM_NODES * $NUM_TRAINING_WEEKS * ($INTRA_CLUSTER_PROB + $INTER_CLUSTER_PROB)" | bc)
                                        RAW_MEM=$(printf "%.0f" "$RAW_MEM")
                                        
                                        for SEED in 3457
                                        do
                                            for NODE_FEAT in "ONE_HOT"
                                            do
                                                # As far as NODE_FEAT=ONE_HOT, it's not important what is the node feature dimension!
                                                for NODE_FEAT_DIM in 1
                                                do
                                                    # DYGLIB training
                                                    for model in "CTDG/_dygformer" "CTDG/_tgn" "CTDG/_tgat" "CTDG/_tgn_tgb"
                                                    do
                                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                                            # Memory computation for methods implemented by DyGLib
                                                            if (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                                                MEM=4
                                                            else
                                                                MEM=$RAW_MEM
                                                            fi
                                                            GPU=40
                                                            MAX_GPU=40

                                                            for NUM_UNITS in 1
                                                            do
                                                                for NUM_HEADS in 2
                                                                do
                                                                    for TIME_FEAT_DIM in 100
                                                                    do
                                                                        for NUM_NEIGHBORS in 20
                                                                        do
                                                                            for CHANNEL_EMBEDDING_DIM in 50
                                                                            do
                                                                                for MAX_INPUT_SEQ_LEN in 4 8 16
                                                                                do
                                                                                    for TRAIN_BATCH_SIZE in 1
                                                                                    do
                                                                                        for MEMORY_DIM in 100
                                                                                        do
                                                                                            sbatch \
                                                                                                --mem=${MEM}gb \
                                                                                                --gres=gpu:1 \
                                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                                                            $task/$model.sh \
                                                                                                    "$DATA" \
                                                                                                    $SEED \
                                                                                                    $NODE_FEAT \
                                                                                                    $NODE_FEAT_DIM \
                                                                                                    $EVAL_MODE \
                                                                                                    $NUM_EPOCHS_TO_VIS \
                                                                                                    $ROOT_LOAD_SAVE_DIR \
                                                                                                    "$VAL_FIRST_METRIC" \
                                                                                                    $MEM \
                                                                                                    $MAX_GPU \
                                                                                                    $GPU \
                                                                                                    $NUM_UNITS \
                                                                                                    $NUM_HEADS \
                                                                                                    $TIME_FEAT_DIM \
                                                                                                    $NUM_NEIGHBORS \
                                                                                                    $TRAIN_BATCH_SIZE \
                                                                                                    $CTDG_DO_SNAPSHOT_TRAINING \
                                                                                                    $CHANNEL_EMBEDDING_DIM \
                                                                                                    $MAX_INPUT_SEQ_LEN \
                                                                                                    $MEMORY_DIM \
                                                                                                    $CLEAR_RESULT \
                                                                                                    $WANDB_ENTITY
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

                                                    # CTDG/_ctan
                                                    if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_ctan " ]]; then
                                                        # Memory computation for CTAN
                                                        if (( $(echo "$RAW_MEM > 40" | bc -l) )); then
                                                            MEM=40
                                                        elif (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                                            MEM=4
                                                        else
                                                            MEM=$RAW_MEM
                                                        fi
                                                        
                                                        GPU=40
                                                        MAX_GPU=40

                                                        for NUM_UNITS in 1 
                                                        do
                                                            for OUT_CHANNELS in 128
                                                            do
                                                                for TIME_FEAT_DIM in 100
                                                                do
                                                                    for SAMPLER_SIZE in 32 64 128
                                                                    do
                                                                        for TRAIN_BATCH_SIZE in 1
                                                                        do
                                                                            sbatch \
                                                                                --mem=${MEM}gb \
                                                                                --gres=gpu:1 \
                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/CTDG/_ctan/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/CTDG/_ctan/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                                                $task/CTDG/_ctan.sh \
                                                                                    "$DATA" \
                                                                                    $SEED \
                                                                                    $NODE_FEAT \
                                                                                    $NODE_FEAT_DIM \
                                                                                    $EVAL_MODE \
                                                                                    $NUM_EPOCHS_TO_VIS \
                                                                                    $ROOT_LOAD_SAVE_DIR \
                                                                                    "$VAL_FIRST_METRIC" \
                                                                                    $MEM \
                                                                                    $MAX_GPU \
                                                                                    $GPU \
                                                                                    $NUM_UNITS \
                                                                                    $OUT_CHANNELS \
                                                                                    $TIME_FEAT_DIM \
                                                                                    $SAMPLER_SIZE \
                                                                                    $TRAIN_BATCH_SIZE \
                                                                                    $CTDG_DO_SNAPSHOT_TRAINING \
                                                                                    $CLEAR_RESULT \
                                                                                    $WANDB_ENTITY
                                                                        done
                                                                    done
                                                                done
                                                            done
                                                        done
                                                    fi
                                                    
                                                    # Discrete-time dynamic graph methods
                                                    for model in "DTDG/_gcn" "DTDG/_gclstm" "DTDG/_egcno" "DTDG/_tgcn" "DTDG/_gat"
                                                    do
                                                        for NUM_UNITS in 1
                                                        do
                                                            for OUT_CHANNELS in 128
                                                            do
                                                                if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                                                    if (( $(echo "$RAW_MEM > 16" | bc -l) )); then
                                                                        MEM=16
                                                                    elif (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                                                        MEM=4
                                                                    else
                                                                        MEM=$RAW_MEM
                                                                    fi
                                                                    sbatch \
                                                                        --mem=${MEM}gb \
                                                                        --gres=gpu:1 \
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/${SEED}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                                        $task/$model.sh \
                                                                            "$DATA" \
                                                                            $SEED \
                                                                            $NODE_FEAT \
                                                                            $NODE_FEAT_DIM \
                                                                            $EVAL_MODE \
                                                                            $NUM_EPOCHS_TO_VIS \
                                                                            $ROOT_LOAD_SAVE_DIR \
                                                                            "$VAL_FIRST_METRIC" \
                                                                            $OUT_CHANNELS \
                                                                            $NUM_UNITS \
                                                                            $CLEAR_RESULT \
                                                                            $WANDB_ENTITY
                                                                fi
                                                            done
                                                        done
                                                    done
                                                    # Baseline models
                                                    for model in "DTDG/_empty" "DTDG/_clique" "DTDG/_previous"
                                                    do
                                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                                            sbatch \
                                                                --mem=4gb \
                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-o.out" \
                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_${dataset_pattern_indices}/${model}/slurm-${NUM_TRAINING_WEEKS}trW-nc${NUM_CLUSTERS}-icp${INTRA_CLUSTER_PROB}-incp${INTER_CLUSTER_PROB}-%j-e.out" \
                                                                $task/$model.sh \
                                                                "$DATA" \
                                                                $ROOT_LOAD_SAVE_DIR \
                                                                $WANDB_ENTITY
                                                        fi
                                                    done
                                                done
                                            done
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
done
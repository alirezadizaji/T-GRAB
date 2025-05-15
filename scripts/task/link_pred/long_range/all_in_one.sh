cd $HOME/lab/TSA/scripts/task/link_pred/
task=long_range
NUM_EPOCHS_TO_VIS=0
ROOT_LOAD_SAVE_DIR=$SCRATCH
which_dataset_to_train=("$@")
DYGLIB=("CTDG/_dygformer" "CTDG/_tgn" "CTDG/_tgat")

# User can change these variables.
EVAL_MODE=false
CTDG_DO_SNAPSHOT_TRAINING=false
METHODS_TO_RUN=("CTDG/_ctan")
CLEAR_RESULT=false

for value in "${which_dataset_to_train[@]}"; 
do
    ## sbm stochastic training
    if [[ "$value" == "er" ]]; then
        VAL_FIRST_METRIC="day0_avg_f1"
        NUM_NODES=100
        PATTERN_MODE="er"

        #Periodicity training
        for VAL_RATIO in 0.1
        do
            for TEST_RATIO in 0.1
            do
                for ER_D in 0.4
                do
                    for ER_R in 0.01
                    do
                        for K in 128 256 512
                        do
                            for N in 100 200
                            do
                                DATA="($K, $N)/long_range-${NUM_NODES}n-erpm-${VAL_RATIO}vr-${TEST_RATIO}tr-${ER_R}eps-${ER_D}epp"
                                
                                if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                    # Edgebank doesn't need seed, or node_feat
                                    sbatch \
                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($K, $N)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${ER_R}eps-${ER_D}epp-o.out" \
                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($K, $N)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${ER_R}eps-${ER_D}epp-e.out" \
                                        $task/CTDG/_edgebank.sh \
                                            "$DATA" \
                                            $ROOT_LOAD_SAVE_DIR \
                                            "$VAL_FIRST_METRIC"
                                fi

                                # Compute memory
                                # Following formula was found empirically to avoid oom in all cases.
                                RAW_MEM=$(echo "1 * $K * $N * $ER_R" | bc)
                                RAW_MEM=$(printf "%.0f" "$RAW_MEM")

                                for SEED in 1235
                                do
                                    for NODE_FEAT in "ONE_HOT"
                                    do
                                        # As far as NODE_FEAT=ONE_HOT, it's not important what is the node feature dimension!
                                        for NODE_FEAT_DIM in 1
                                        do
                                            # DYGLIB training
                                            for model in "CTDG/_dygformer" "CTDG/_tgn" "CTDG/_tgat"
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
                                                                        for MAX_INPUT_SEQ_LEN in 256
                                                                        do
                                                                            for TRAIN_BATCH_SIZE in 100
                                                                            do
                                                                                for MEMORY_DIM in 100
                                                                                do
                                                                                    sbatch \
                                                                                        --mem=${MEM}gb \
                                                                                        --gres=gpu:1 \
                                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($K, $N)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${ER_R}eps-${ER_D}epp-o.out" \
                                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($K, $N)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${ER_R}eps-${ER_D}epp-e.out" \
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
                                                                                            $CLEAR_RESULT
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
                                                                for TRAIN_BATCH_SIZE in 100
                                                                do
                                                                    sbatch \
                                                                        --mem=${MEM}gb \
                                                                        --gres=gpu:1 \
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($K, $N)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${ER_R}eps-${ER_D}epp-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($K, $N)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${ER_R}eps-${ER_D}epp-e.out" \
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
                                                                            $CLEAR_RESULT
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
                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($K, $N)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${ER_R}eps-${ER_D}epp-o.out" \
                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($K, $N)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${ER_R}eps-${ER_D}epp-e.out" \
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
                                                                    $CLEAR_RESULT
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
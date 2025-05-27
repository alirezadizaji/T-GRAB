export PROJ_ROOT_DIR="$HOME/lab"
export ROOT_LOAD_SAVE_DIR="$SCRATCH/"
export SCRIPT_LOC=scripts/task/link_pred/
export DATA_LOC=lab/TSA/data/
export PYENV=$HOME/envs/tsa/

cd $PROJ_ROOT_DIR/TSA/$SCRIPT_LOC

task=memory_node
NUM_EPOCHS_TO_VIS=0
which_dataset_to_train=("$@")
DYGLIB=("CTDG/_dygformer" "CTDG/_tgn" "CTDG/_tgat")

# User can change these variables.
EVAL_MODE=false
CTDG_DO_SNAPSHOT_TRAINING=true
# METHODS_TO_RUN=("CTDG/_edgebank" "CTDG/_dygformer" "CTDG/_ctan" "CTDG/_tgn" "CTDG/_tgat" "DTDG/_gcn" "DTDG/_gclstm" "DTDG/_egcno" "DTDG/_tgcn" "DTDG/_gat" "DTDG/_egcnh" "DTDG/_clique" "DTDG/_previous")
METHODS_TO_RUN=("CTDG/_dygformer")
CLEAR_RESULT=false
WANDB_ENTITY="##anonymized##"

for value in "${which_dataset_to_train[@]}"; 
do
    ## sbm stochastic training
    if [[ "$value" == "er" ]]; then
        VAL_FIRST_METRIC="memnode_avg_f1"
        NUM_NODES=101
        PATTERN_MODE="er"
        #Periodicity training
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
                                for GAP in 64 256
                                do
                                    for NUM_PATTERNS in 4000
                                    do
                                        DATA="($GAP, $NUM_PATTERNS)/memory_node-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi"
                                        
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                            # Edgebank doesn't need seed, or node_feat
                                            sbatch \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
                                                $task/CTDG/_edgebank.sh \
                                                    "$DATA" \
                                                    $ROOT_LOAD_SAVE_DIR \
                                                    "$VAL_FIRST_METRIC"
                                        fi

                                        # Compute memory
                                        # Following formula was found empirically to avoid oom in all cases.
                                        RAW_MEM=$(echo "0.00008 * ($GAP + $NUM_PATTERNS) * $NUM_NODES * $NUM_NODES * $ER_PROB" | bc)
                                        RAW_MEM=$(printf "%.0f" "$RAW_MEM")
                                        
                                        for SEED in 3457
                                        do
                                            for NODE_FEAT in "ONE_HOT"
                                            do
                                                # As far as NODE_FEAT=ONE_HOT, it's not important what is the node feature dimension!
                                                for NODE_FEAT_DIM in 1
                                                do
                                                    # DYGLIB/TGB training
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
                                                                                    if [[ "${model}" = "CTDG/_dygformer" ]]; then
                                                                                        NUM_HEADS=2
                                                                                    else
                                                                                        NUM_HEADS=3
                                                                                    fi
                                                                                    # NUM_HEADS=2
                                                                                    sbatch \
                                                                                        --mem=${MEM}gb \
                                                                                        --gres=gpu:1 \
                                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                            $CLEAR_RESULT \
                                                                                            $NUM_NODES \
                                                                                            $WANDB_ENTITY
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
                                                                    for SAMPLER_SIZE in 32 64 128 256 512
                                                                    do
                                                                        for TRAIN_BATCH_SIZE in 1
                                                                        do
                                                                            sbatch \
                                                                                --mem=${MEM}gb \
                                                                                --gres=gpu:1 \
                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                    $NUM_NODES \
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
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                            $NUM_NODES \
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
                    done
                done
            done
        done
    fi

    ############################################################
    # SCEU
    ############################################################
    if [[ "$value" == "SCEU" ]]; then
        VAL_FIRST_METRIC="memnode_avg_f1"
        NUM_NODES=100
        PATTERN_MODE="uniform"
        #Periodicity training
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
                                for GAP in 1
                                do
                                    for DIST in 1 3 15 63 255
                                    do
                                        ALL_NODES_NUM=$((NUM_NODES + DIST + 1))
                                        DATA="($GAP, $DIST)/SLRCE-4000ns-${ALL_NODES_NUM}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei"
                                        
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                            # Edgebank doesn't need seed, or node_feat
                                            sbatch \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-e.out" \
                                                $task/CTDG/_edgebank.sh \
                                                    "$DATA" \
                                                    $ROOT_LOAD_SAVE_DIR \
                                                    "$VAL_FIRST_METRIC"
                                        fi

                                        # Compute memory
                                        # Following formula was found empirically to avoid oom in all cases.
                                        RAW_MEM=$(echo "0.0000012 * ($GAP + $DIST) * $NUM_NODES * $NUM_NODES * $NUM_EDGES" | bc)
                                        RAW_MEM=$(printf "%.0f" "$RAW_MEM")
                                        
                                        for SEED in 1235
                                        do
                                            for NODE_FEAT in "ONE_HOT"
                                            do
                                                # As far as NODE_FEAT=ONE_HOT, it's not important what is the node feature dimension!
                                                for NODE_FEAT_DIM in 1
                                                do
                                                    # DYGLIB/TGB training
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
                                                                                    NUM_HEADS=2
                                                                                    sbatch \
                                                                                        --mem=${MEM}gb \
                                                                                        --gres=gpu:1 \
                                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-o.out" \
                                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-e.out" \
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
                                                                                            $CLEAR_RESULT \
                                                                                            $ALL_NODES_NUM \
                                                                                            $WANDB_ENTITY
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
                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/CTDG/_ctan/slurm-%j-${ALL_NODES_NUM}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-o.out" \
                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/CTDG/_ctan/slurm-%j-${ALL_NODES_NUM}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-e.out" \
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
                                                                                    $ALL_NODES_NUM \
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
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/${model}/slurm-%j-${ALL_NODES_NUM}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/${model}/slurm-%j-${ALL_NODES_NUM}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-e.out" \
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
                                                                            $ALL_NODES_NUM \
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
                                                                --mem=4g \
                                                                --partition=long-cpu \
                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/${model}/slurm-%j-${ALL_NODES_NUM}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-o.out" \
                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $DIST)/${model}/slurm-%j-${ALL_NODES_NUM}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${NUM_EDGES}ne-${NUM_EDGES_INDUCTIVE}nei-e.out" \
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

    ############################################################
    ## SCEU V3
    ############################################################
    if [[ "$value" == "SCEU_V3" ]]; then
        VAL_FIRST_METRIC="memnode_avg_f1"
        NUM_BRANCHES=3
        NUM_NODES=100
        #Periodicity training
        for VAL_RATIO in 0.1
        do
            for TEST_RATIO in 0.1
            do
                for LAG in 32
                do
                    for BRANCH_LEN in 1 2 4 8 16
                    do
                        DATA="($LAG, $BRANCH_LEN)/SLRCE-V3-4000ns-${NUM_NODES}nn-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr"
                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                            # Edgebank doesn't need seed, or node_feat
                            sbatch \
                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/CTDG/_edgebank/slurm-%j-${NUM_NODES}nn-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/CTDG/_edgebank/slurm-%j-${NUM_NODES}nn-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \
                                $task/CTDG/_edgebank.sh \
                                    "$DATA" \
                                    $ROOT_LOAD_SAVE_DIR \
                                    "$VAL_FIRST_METRIC"
                        fi

                        # Compute memory
                        # Following formula was found empirically to avoid oom in all cases.
                        RAW_MEM=$(echo "0.05 * $NUM_BRANCHES * $BRANCH_LEN" | bc)
                        RAW_MEM=$(printf "%.0f" "$RAW_MEM")

                        for SEED in 1235 2346 3457
                        do
                            for NODE_FEAT in "ONE_HOT"
                            do
                                # As far as NODE_FEAT=ONE_HOT, it's not important what is the node feature dimension!
                                for NODE_FEAT_DIM in 1
                                do
                                    # DYGLIB/TGB training
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
                                                                    NUM_HEADS=2
                                                                    sbatch \
                                                                        --mem=${MEM}gb \
                                                                        --gres=gpu:1 \
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \
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
                                                                                $CLEAR_RESULT \
                                                                                $NUM_NODES \
                                                                                $WANDB_ENTITY
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
                                                                --time=1-12:00:00 \
                                                                --account=def-bengioy \
                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/CTDG/_ctan/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/CTDG/_ctan/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \
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
                                                                    $NUM_NODES \
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
                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \
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
                                                            $NUM_NODES \
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
                                                --mem=4g \
                                                --time=00:20:00 \
                                                --account=def-bengioy \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \
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


            ############################################################
    # SCEU V2
    ############################################################
    if [[ "$value" == "SCEU_V2" ]]; then
        VAL_FIRST_METRIC="memnode_avg_f1"
        NUM_BRANCHES=100

        #Periodicity training
        for VAL_RATIO in 0.1
        do
            for TEST_RATIO in 0.1
            do
                for NUM_ACTIVE_BRANCHES in 20
                do
                    for LAG in 1 4 16
                    do
                        for BRANCH_LEN in 1 2 4 8 16
                        do
                            DATA="($LAG, $BRANCH_LEN)/SLRCE-V2-star-4000ns-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr"
                            if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                # Edgebank doesn't need seed, or node_feat
                                sbatch \
                                    --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/CTDG/_edgebank/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                    --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/CTDG/_edgebank/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \

                                    $task/CTDG/_edgebank.sh \
                                        "$DATA" \
                                        $ROOT_LOAD_SAVE_DIR \
                                        "$VAL_FIRST_METRIC"
                            fi

                            # Compute memory
                            # Following formula was found empirically to avoid oom in all cases.
                            RAW_MEM=$(echo "0.05 * $NUM_ACTIVE_BRANCHES * $BRANCH_LEN" | bc)
                            RAW_MEM=$(printf "%.0f" "$RAW_MEM")

                            for SEED in 1235
                            do
                                for NODE_FEAT in "ONE_HOT"
                                do
                                    # As far as NODE_FEAT=ONE_HOT, it's not important what is the node feature dimension!
                                    for NODE_FEAT_DIM in 1
                                    do
                                        # DYGLIB/TGB training
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
                                                                        NUM_HEADS=2
                                                                        sbatch \
                                                                            --mem=${MEM}gb \
                                                                            --gres=gpu:1 \
                                                                            --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                                                            --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \
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
                                                                                    $CLEAR_RESULT \
                                                                                    $ALL_NODES_NUM \
                                                                                    $WANDB_ENTITY
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
                                                                    --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/CTDG/_ctan/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                                                    --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/CTDG/_ctan/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \
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
                                                                        $ALL_NODES_NUM \
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
                                                            --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                                            --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \
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
                                                                $ALL_NODES_NUM \
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
                                                    --mem=4g \
                                                    --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-o.out" \
                                                    --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($LAG, $BRANCH_LEN)/${model}/slurm-%j-${NUM_ACTIVE_BRANCHES}anb-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr-e.out" \
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
    fi

    if [[ "$value" == "er_t_unit" ]]; then
        VAL_FIRST_METRIC="memnode_avg_f1"
        NUM_NODES=101
        PATTERN_MODE="er"
        #Periodicity training
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
                                        DATA="($GAP, $NUM_PATTERNS)/memory_node4t-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi"
                                        
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                            # Edgebank doesn't need seed, or node_feat
                                            sbatch \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
                                                $task/CTDG/_edgebank.sh \
                                                    "$DATA" \
                                                    $ROOT_LOAD_SAVE_DIR \
                                                    "$VAL_FIRST_METRIC"
                                        fi

                                        # Compute memory
                                        # Following formula was found empirically to avoid oom in all cases.
                                        RAW_MEM=$(echo "0.00008 * ($GAP + $NUM_PATTERNS) * $NUM_NODES * $NUM_NODES * $ER_PROB" | bc)
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
                                                                                    if [[ "${model}" = "CTDG/_dygformer" ]]; then
                                                                                        NUM_HEADS=2
                                                                                    else
                                                                                        NUM_HEADS=3
                                                                                    fi
                                                                                    # NUM_HEADS=2
                                                                                    sbatch \
                                                                                        --mem=${MEM}gb \
                                                                                        --gres=gpu:1 \
                                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                            $CLEAR_RESULT \
                                                                                            $NUM_NODES \
                                                                                            $WANDB_ENTITY
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
                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                    $NUM_NODES \
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
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                            $NUM_NODES \
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
                    done
                done
            done
        done
    fi


    ## sbm stochastic training
    if [[ "$value" == "nonlinear_er" ]]; then
        VAL_FIRST_METRIC="memnode_avg_f1"
        NUM_NODES=1001
        PATTERN_MODE="nonlinear_er"

        #Periodicity training
        for VAL_RATIO in 0.1
        do
            for TEST_RATIO in 0.1
            do
                for TEST_INDUCTIVE_RATIO in 0.1
                do
                    for TEST_INDUCTIVE_NUM_NODES_RATIO in 0.1
                    do
                        for ER_PROB in 0.0001
                        do
                            for ER_PROB_INDUCTIVE in 0.001
                            do
                                for GAP in 1 2 3 4
                                do
                                    for NUM_PATTERNS in 1000 2000 4000 8000 16000 32000
                                    do
                                        DATA="($GAP, $NUM_PATTERNS)/memory_node-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-1.0mean-1.0std"
                                        
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                            # Edgebank doesn't need seed, or node_feat
                                            sbatch \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
                                                $task/CTDG/_edgebank.sh \
                                                    "$DATA" \
                                                    $ROOT_LOAD_SAVE_DIR \
                                                    "$VAL_FIRST_METRIC"
                                        fi

                                        # Compute memory
                                        # Following formula was found empirically to avoid oom in all cases.
                                        RAW_MEM=$(echo "0.00008 * ($GAP + $NUM_PATTERNS) * $NUM_NODES * $NUM_NODES * $ER_PROB" | bc)
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
                                                                                    if [[ "${model}" = "CTDG/_dygformer" ]]; then
                                                                                        NUM_HEADS=2
                                                                                    else
                                                                                        NUM_HEADS=3
                                                                                    fi
                                                                                    sbatch \
                                                                                        --mem=${MEM}gb \
                                                                                        --gres=gpu:1 \
                                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                            $CLEAR_RESULT \
                                                                                            $NUM_NODES \
                                                                                            $WANDB_ENTITY
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
                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                    $NUM_NODES \
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
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                            $NUM_NODES \
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
            done
        done
    fi

    ## sbm stochastic training
    if [[ "$value" == "nonlin_er_const_w_had" ]]; then
        VAL_FIRST_METRIC="memnode_avg_f1"
        NUM_NODES=101
        PATTERN_MODE="nonlin_er_const_w_had"

        #Periodicity training
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
                                for GAP in 1 4
                                do
                                    for NUM_PATTERNS in 1000 2000 4000 8000 16000 32000
                                    do
                                        DATA="($GAP, $NUM_PATTERNS)/memory_node-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-0.0mean-1.0std-5643wseed"
                                        
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                            # Edgebank doesn't need seed, or node_feat
                                            sbatch \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
                                                $task/CTDG/_edgebank.sh \
                                                    "$DATA" \
                                                    $ROOT_LOAD_SAVE_DIR \
                                                    "$VAL_FIRST_METRIC"
                                        fi

                                        # Compute memory
                                        # Following formula was found empirically to avoid oom in all cases.
                                        RAW_MEM=$(echo "0.00008 * ($GAP + $NUM_PATTERNS) * $NUM_NODES * $NUM_NODES * $ER_PROB" | bc)
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
                                                                                    if [[ "${model}" = "CTDG/_dygformer" ]]; then
                                                                                        NUM_HEADS=2
                                                                                    else
                                                                                        NUM_HEADS=3
                                                                                    fi
                                                                                     sbatch \
                                                                                         --mem=${MEM}gb \
                                                                                         --gres=gpu:1 \
                                                                                         --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                         --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                            $CLEAR_RESULT \
                                                                                            $NUM_NODES \
                                                                                            $WANDB_ENTITY
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
                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                    $NUM_NODES \
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
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                            $NUM_NODES \
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
            done
        done
    fi

    ## sbm stochastic training
    if [[ "$value" == "nonlin_er_const_w_dot" ]]; then
        VAL_FIRST_METRIC="memnode_avg_f1"
        NUM_NODES=1001
        PATTERN_MODE="nonlin_er_const_w_dot"

        #Periodicity training
        for VAL_RATIO in 0.1
        do
            for TEST_RATIO in 0.1
            do
                for TEST_INDUCTIVE_RATIO in 0.1
                do
                    for TEST_INDUCTIVE_NUM_NODES_RATIO in 0.1
                    do
                        for ER_PROB in 0.0001
                        do
                            for ER_PROB_INDUCTIVE in 0.001
                            do
                                for GAP in 1 4
                                do
                                    for NUM_PATTERNS in 16000 32000
                                    do
                                        DATA="($GAP, $NUM_PATTERNS)/memory_node-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-0.0mean-1.0std-5643wseed"
                                        
                                        if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                            # Edgebank doesn't need seed, or node_feat
                                            sbatch \
                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
                                                $task/CTDG/_edgebank.sh \
                                                    "$DATA" \
                                                    $ROOT_LOAD_SAVE_DIR \
                                                    "$VAL_FIRST_METRIC"
                                        fi

                                        # Compute memory
                                        # Following formula was found empirically to avoid oom in all cases.
                                        RAW_MEM=$(echo "0.00008 * ($GAP + $NUM_PATTERNS) * $NUM_NODES * $NUM_NODES * $ER_PROB" | bc)
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
                                                                                    if [[ "${model}" = "CTDG/_dygformer" ]]; then
                                                                                        NUM_HEADS=2
                                                                                    else
                                                                                        NUM_HEADS=3
                                                                                    fi
                                                                                    sbatch \
                                                                                        --mem=${MEM}gb \
                                                                                        --gres=gpu:1 \
                                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                            $CLEAR_RESULT \
                                                                                            $NUM_NODES \
                                                                                            $WANDB_ENTITY
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
                                                                                --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                                    $NUM_NODES \
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
                                                                        --output="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                        --error="${SCRATCH}/lab/TSA/scripts/tasks/logs/${task}/${value}_($GAP, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
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
                                                                            $NUM_NODES \
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
            done
        done
    fi
done

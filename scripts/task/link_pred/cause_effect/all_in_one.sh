export ROOT_LOAD_SAVE_DIR="$PWD/scratch/"
export SCRIPT_LOC=scripts/task/link_pred/
export DATA_LOC=$PWD/data/
export PYENV=$PWD/tgrab/

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

task=cause_effect
NUM_EPOCHS_TO_VIS=0

###################### Running-specific variables #########################
EVAL_MODE=false
CTDG_DO_SNAPSHOT_TRAINING=true
# METHODS_TO_RUN=("CTDG/_edgebank" "CTDG/_dygformer" "CTDG/_ctan" "CTDG/_tgn" "CTDG/_tgat" "DTDG/_gcn" "DTDG/_gclstm" "DTDG/_egcn" "DTDG/_tgcn" "DTDG/_gat" "DTDG/_previous")
METHODS_TO_RUN=("DTDG/_tgcn")
CLEAR_RESULT=false
WANDB_ENTITY="alirezadizaji24-universit-de-montr-al"
###########################################################################

VAL_FIRST_METRIC="memnode_avg_f1"

#@@@@@@@@@@@@@@@@@@@@@ Dataset-specific variables @@@@@@@@@@@@@@@@@@@@@@@@@@
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
                        for LAG in 1
                        do
                            for NUM_PATTERNS in 4000
                            do
                                DATA="($LAG, $NUM_PATTERNS)/cause_effect-${NUM_NODES}n-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi"
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#$$$$$$$$$$$$$$$$$$$$$$$$$ Model-specific variables $$$$$$$$$$$$$$$$$$$$$$$$$$$
                                if [[ " ${METHODS_TO_RUN[@]} " =~ " CTDG/_edgebank " ]]; then
                                    # Edgebank doesn't need seed, or node_feat
                                    sbatch \
                                        --output="$PWD/scripts/tasks/logs/${task}/${value}_($LAG, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                        --error="$PWD/scripts/tasks/logs/${task}/${value}_($LAG, $NUM_PATTERNS)/CTDG/_edgebank/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
                                        scripts/task/link_pred/cause_effect/CTDG/_edgebank.sh \
                                            "$DATA" \
                                            $ROOT_LOAD_SAVE_DIR \
                                            "$VAL_FIRST_METRIC"
                                fi

                                # Compute memory
                                # Following formula was found empirically to avoid oom in all cases.
                                RAW_MEM=$(echo "0.00008 * ($LAG + $NUM_PATTERNS) * $NUM_NODES * $NUM_NODES * $ER_PROB" | bc)
                                RAW_MEM=$(printf "%.0f" "$RAW_MEM")
                                
                                for SEED in 3457
                                do
                                    for NODE_FEAT in "ONE_HOT"
                                    do
                                        # As far as NODE_FEAT=ONE_HOT, it's not important what is the node feature dimension!
                                        for NODE_FEAT_DIM in 1
                                        do
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
                                                                                NUM_HEADS=3 #  To enable the sum of node_feat_dim and time_feat_dim be divided by num_heads!
                                                                            fi
                                                                            sbatch \
                                                                                --mem=${MEM}gb \
                                                                                --gres=gpu:1 \
                                                                                --output="$PWD/scripts/tasks/logs/${task}/${value}_($LAG, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                                --error="$PWD/scripts/tasks/logs/${task}/${value}_($LAG, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
                                                                                scripts/task/link_pred/cause_effect/$model.sh \
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
                                                                        --output="$PWD/scripts/tasks/logs/${task}/${value}_($LAG, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                        --error="$PWD/scripts/tasks/logs/${task}/${value}_($LAG, $NUM_PATTERNS)/CTDG/_ctan/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
                                                                        scripts/task/link_pred/cause_effect/CTDG/_ctan.sh \
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
                                            for model in "DTDG/_gcn" "DTDG/_gclstm" "DTDG/_egcn" "DTDG/_tgcn" "DTDG/_gat"
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
                                                                --output="$PWD/scripts/tasks/logs/${task}/${value}_($LAG, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-o.out" \
                                                                --error="$PWD/scripts/tasks/logs/${task}/${value}_($LAG, $NUM_PATTERNS)/${model}/slurm-%j-${NUM_NODES}n-${PATTERN_MODE}pm-${VAL_RATIO}vr-${TEST_RATIO}tr-${TEST_INDUCTIVE_RATIO}tir-${TEST_INDUCTIVE_NUM_NODES_RATIO}tinnr-${ER_PROB}ep-${ER_PROB_INDUCTIVE}epi-e.out" \
                                                                scripts/task/link_pred/cause_effect/$model.sh \
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
                                            for model in "DTDG/_previous"
                                            do
                                                if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                                    sbatch \
                                                        --mem=4g \
                                                        --partition=long-cpu \
                                                        --output="$PWD/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${model}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-%j-o.out" \
                                                        --error="$PWD/scripts/tasks/logs/${task}/${value}_${DATASET_PATTERN}/${model}/slurm-${NUM_TRAINING_WEEKS}trW-${EVAL_WEEK}vW-${EVAL_WEEK}tsW-%j-e.out" \
                                                        scripts/task/link_pred/cause_effect/$model.sh \
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
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    done
done

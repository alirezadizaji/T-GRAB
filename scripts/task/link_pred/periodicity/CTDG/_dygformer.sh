#!/bin/bash
#SBATCH --job-name=CT_Pe_dygformer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

DATA_LOC=data/
RUN_SCRIPT=T-GRAB.train.run
NODE_POS=circular_layout

# Load module, env
module load python/3.8
source $PWD/tgrab/bin/activate
cd ../

DATA="$1"
SEED=$2
NODE_FEAT=$3
NODE_FEAT_DIM=$4
EVAL_MODE=$5
NUM_EPOCHS_TO_VIS=$6
ROOT_LOAD_SAVE_DIR=$7
VAL_FIRST_METRIC=${8}

NUM_UNITS=1
echo "@@@ RUNNING DyGFormer on $DATA @@@"
echo "^^^ Number of units: $NUM_UNITS; ^^^"

MAX_BATCH_SIZE=5000
GPU=${11}
MAX_GPU=${10}

BATCH_SIZE=$((MAX_BATCH_SIZE * GPU / MAX_GPU))
BATCH_SIZE=$(printf "%.0f" "$BATCH_SIZE")

NUM_UNITS=${12}
NUM_HEADS=${13}
TIME_FEAT_DIM=${14}
NUM_NEIGHBORS=${15}
TRAIN_BATCH_SIZE=${16}
TRAIN_SNAPSHOT_BASED=${17}
CHANNEL_EMBEDDING_DIM=${18}
MAX_INPUT_SEQ_LEN=${19}
CLEAR_RESULT=${21}
WANDB_ENTITY=${22}
ARGS=(
    CTDG.link_pred.periodicity.dygformer
    --data="$DATA"
    --seed=$SEED
    --patience=100
    --num-epoch=100000
    --node-feat=$NODE_FEAT
    --data-loc=$DATA_LOC
    --num-units=$NUM_UNITS
    --val-first-metric=$VAL_FIRST_METRIC
    --node-pos=$NODE_POS
    --node-feat-dim=$NODE_FEAT_DIM
    # --train-batch-size=$BATCH_SIZE
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR
    --time-scaling-factor=0.000001
    --num-units=$NUM_UNITS
    --num-heads=$NUM_HEADS 
    --dropout=0.1 
    --time-feat-dim=$TIME_FEAT_DIM
    --patch_size=8
    --train-eval-gap=8
    --channel_embedding_dim=$CHANNEL_EMBEDDING_DIM
    --max_input_sequence_length=$MAX_INPUT_SEQ_LEN
    --train-batch-size=$TRAIN_BATCH_SIZE
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project="T-GRAB" \
    --wandb-log-interval=1 # DyGFormer is a very slow model. So, we need to log every 1 step.
)

# Training arguments
TRAIN_ARGS=(
    "${ARGS[@]}" 
    "--num-epochs-to-visualize=$NUM_EPOCHS_TO_VIS"
)
# Evaluation arguments
EVAL_ARGS=(
    "${ARGS[@]}" 
    "--num-epochs-to-visualize=0" 
    "--eval-mode"
)

if [ "$CLEAR_RESULT" == "true" ]; then
    TRAIN_ARGS=(
        "${TRAIN_ARGS[@]}"
        --clear-results
    )
fi
if [ "$TRAIN_SNAPSHOT_BASED" == "true" ]; then
    TRAIN_ARGS=(
        "${TRAIN_ARGS[@]}"
        "--train-snapshot-based"
    )
    EVAL_ARGS=(
        "${EVAL_ARGS[@]}"
        "--train-snapshot-based"
    )
fi

# Training
if [ "$EVAL_MODE" == "false" ]; then
    echo -e "\n\n %% START TRAINING... %%"
    python -m $RUN_SCRIPT "${TRAIN_ARGS[@]}"
else
    # Evaluation: to visualize the model output for the best epoch
    echo -e "\n\n %% START EVALUATION... %%"
    python -m $RUN_SCRIPT "${EVAL_ARGS[@]}"
fi

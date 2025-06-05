#!/bin/bash
#SBATCH --job-name=CT_CE_CTAN
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

echo "@@@ RUNNING CTAN on $DATA @@@"
echo "^^^ Number of units: $NUM_UNITS; number of embedding dim: $OUT_CHANNELS; ^^^"

MAX_BATCH_SIZE=20000
GPU=${11}
MAX_GPU=${10}
BATCH_SIZE=$((MAX_BATCH_SIZE * GPU / MAX_GPU))
BATCH_SIZE=$(printf "%.0f" "$BATCH_SIZE")

NUM_UNITS=${12}
OUT_CHANNELS=${13}
TIME_FEAT_DIM=${14}
SAMPLER_SIZE=${15}
TRAIN_BATCH_SIZE=${16}
TRAIN_SNAPSHOT_BASED=${17}
CLEAR_RESULT=${18}
NUM_NODES=${19}
WANDB_ENTITY=${20}
ARGS=(
    CTDG.link_pred.memory_node.ctan
    --data="$DATA"
    --seed=$SEED
    --node-feat=$NODE_FEAT
    --data-loc=$DATA_LOC
    --num-units=$NUM_UNITS
    --val-first-metric=$VAL_FIRST_METRIC
    --node-pos=$NODE_POS
    --node-feat-dim=$NODE_FEAT_DIM
    --train-eval-gap=2
    --patience=100
    --num-epoch=100000
    # --train-batch-size=$BATCH_SIZE
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR
    --embedding-dim=$OUT_CHANNELS
    --activation-layer=tanh
    --time-feat-dim=$TIME_FEAT_DIM
    --epsilon=0.010001
    --gamma=0.01
    --mean-delta-t=0.
    --std-delta-t=1.
    --init-time=0
    --sampler-size=$SAMPLER_SIZE
    --train-batch-size=$TRAIN_BATCH_SIZE
    --wandb-entity=$WANDB_ENTITY
    --wandb-project="T-GRAB"
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

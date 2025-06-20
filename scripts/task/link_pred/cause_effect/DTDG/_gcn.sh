#!/bin/bash
#SBATCH --job-name=DT_CE_GCN
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

# GCN scripts
DATA=$1
SEED=$2
NODE_FEAT=$3
NODE_FEAT_DIM=$4
EVAL_MODE=$5
NUM_EPOCHS_TO_VIS=$6
ROOT_LOAD_SAVE_DIR=$7
VAL_FIRST_METRIC=${8}

NUM_UNITS=${10}
OUT_CHANNELS=${9}
CLEAR_RESULT=${11}
NUM_NODES=${12}
WANDB_ENTITY=${13}
echo "@@@ RUNNING GCN on $DATA @@@"
echo "^^^ number of channels: $OUT_CHANNELS; Number of layers: $NUM_UNITS ^^^"

ARGS=(
    DTDG.link_pred.memory_node.gcn
    --data="$DATA"
    --seed=$SEED
    --patience=100
    --num-epoch=100000
    --node-feat=$NODE_FEAT
    --data-loc=$DATA_LOC
    --val-first-metric=$VAL_FIRST_METRIC
    --out-channels=$OUT_CHANNELS
    --num-units=$NUM_UNITS
    --node-pos=$NODE_POS
    --node-feat-dim=$NODE_FEAT_DIM
    --back-prop-window-size=1
    --loss-computation=backward_only_last
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR 
    --wandb-entity=$WANDB_ENTITY
    --wandb-project="T-GRAB"
)

# Training arguments
TRAIN_ARGS=(
    "${ARGS[@]}" 
    "--num-epochs-to-visualize=$NUM_EPOCHS_TO_VIS"
)
if [ "$CLEAR_RESULT" == "true" ]; then
    TRAIN_ARGS=(
        "${TRAIN_ARGS[@]}"
        --clear-results
    )
fi

# Evaluation arguments
EVAL_ARGS=(
    "${ARGS[@]}" 
    "--num-epochs-to-visualize=0" 
    "--eval-mode"
)

# Training
if [ "$EVAL_MODE" == "false" ]; then
    echo -e "\n\n %% START TRAINING... %%"
    python -m $RUN_SCRIPT "${TRAIN_ARGS[@]}"
else
    # Evaluation: to visualize the model output for the best epoch
    echo -e "\n\n %% START EVALUATION... %%"
    python -m $RUN_SCRIPT "${EVAL_ARGS[@]}"
fi
#!/bin/bash
#SBATCH --job-name=CT_LR_TGN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

DATA_LOC=lab/TSA/data/
RUN_SCRIPT=TSA.train.run
NODE_POS=circular_layout

# Load module, env
module load python/3.8
source $HOME/envs/tsa/bin/activate
cd $HOME/lab

DATA="$1"
SEED=$2
NODE_FEAT=$3
NODE_FEAT_DIM=$4
EVAL_MODE=$5
NUM_EPOCHS_TO_VIS=$6
ROOT_LOAD_SAVE_DIR=$7
VAL_FIRST_METRIC=${8}
MEM=${9}

NUM_UNITS=1
echo "@@@ RUNNING TGN on $DATA @@@"
echo "^^^ Number of units: $NUM_UNITS; ^^^"

MAX_BATCH_SIZE=10000
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
MEMORY_DIM=${20}
CLEAR_RESULT=${21}

ARGS=(
    CTDG.link_pred.long_range.tgn
    --data="$DATA"
    --seed=$SEED
    --node-feat=$NODE_FEAT
    --data-loc=$DATA_LOC
    --num-units=$NUM_UNITS
    --val-first-metric=$VAL_FIRST_METRIC
    --node-pos=$NODE_POS
    --node-feat-dim=$NODE_FEAT_DIM
    --patience=50
    --num-epoch=100000
    # --train-batch-size=$BATCH_SIZE
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR
    --num-neighbors=$NUM_NEIGHBORS
    --time-scaling-factor=0.000001
    --num-units=$NUM_UNITS
    --num-heads=$NUM_HEADS
    --dropout=0.1
    --time-feat-dim=$TIME_FEAT_DIM
    --train-batch-size=$TRAIN_BATCH_SIZE
    --memory-dim=$MEMORY_DIM
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
if [ "$TRAIN_SNAPSHOT_BASED" == "true" ]; then
    TRAIN_ARGS=(
        "${TRAIN_ARGS[@]}"
        "--train-snapshot-based"
    )
fi

# Evaluation arguments
EVAL_ARGS=(
    "${ARGS[@]}" 
    "--num-epochs-to-visualize=1" 
    "--eval-mode"
)

# Training
if [ "$EVAL_MODE" == "false" ]; then
    echo -e "\n\n %% START TRAINING... %%"
    python -m $RUN_SCRIPT "${TRAIN_ARGS[@]}"
fi

# Evaluation: to visualize the model output for the best epoch
echo -e "\n\n %% START EVALUATION... %%"
python -m $RUN_SCRIPT "${EVAL_ARGS[@]}"

# Draw the plots
echo -e "\n\n %% DRAW PLOTS... %%"
cd $HOME/lab/TSA/scripts/
./plot/2d/long_range/all_in_one.sh

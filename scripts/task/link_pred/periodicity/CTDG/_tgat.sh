#!/bin/bash
#SBATCH --job-name=CT_Pe_TGAT
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

NUM_UNITS=1
echo "@@@ RUNNING TGAT on $DATA @@@"
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
CLEAR_RESULT=${21}
WANDB_ENTITY=${22}
ARGS=(
    CTDG.link_pred.periodicity.tgat
    --data="$DATA"
    --seed=$SEED
    --patience=100
    --num-epoch=100000
    --train-eval-gap=80 # 2x of eval / training time ratio for sbm_sto_v2 (256, 1) task
    --node-feat=$NODE_FEAT
    --data-loc=$DATA_LOC
    --num-units=$NUM_UNITS
    --val-first-metric=$VAL_FIRST_METRIC
    --node-pos=$NODE_POS
    --node-feat-dim=$NODE_FEAT_DIM
    # --train-batch-size=$BATCH_SIZE
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR
    --num-neighbors=$NUM_NEIGHBORS
    --time-scaling-factor=0.000001
    --num-units=$NUM_UNITS
    --num-heads=$NUM_HEADS 
    --dropout=0.1 
    --time-feat-dim=$TIME_FEAT_DIM
    --train-batch-size=$TRAIN_BATCH_SIZE
    --wandb-entity=$WANDB_ENTITY
    --wandb-project="TSA"
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

# # Draw the plots
# echo -e "\n\n %% DRAW PLOTS... %%"
# cd $HOME/lab/TSA/scripts/
# ./plot/2d/periodicity/all_in_one.sh

#!/bin/bash
#SBATCH --job-name=DT_Pe_GCLSTM
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

NUM_UNITS=${10}
OUT_CHANNELS=${9}
CLEAR_RESULT=${11}
WANDB_ENTITY=${12}
K=2
echo "@@@ RUNNING GCLSTM on $DATA @@@"
echo "^^^ Number of units: $NUM_UNITS; number of channels: $OUT_CHANNELS; Chebyshev filter size: $K ^^^"

ARGS=(
    DTDG.link_pred.periodicity.gclstm
    --data="$DATA"
    --seed=$SEED
    --patience=100
    --num-epoch=100000
    --node-feat=$NODE_FEAT
    --data-loc=$DATA_LOC
    --num-units=$NUM_UNITS
    --val-first-metric=$VAL_FIRST_METRIC
    --out-channels=$OUT_CHANNELS
    --k-gclstm=$K
    --node-pos=$NODE_POS
    --node-feat-dim=$NODE_FEAT_DIM
    --back-prop-window-size=1
    --loss-computation=backward_only_last
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project="TSA"
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

# # Draw the plots
# echo -e "\n\n %% DRAW PLOTS... %%"
# cd $HOME/lab/TSA/scripts/
# ./plot/2d/periodicity/all_in_one.sh
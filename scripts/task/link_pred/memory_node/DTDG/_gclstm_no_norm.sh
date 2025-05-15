#!/bin/bash
#SBATCH --job-name=DT_MN_nn_GCLSTM
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
pwd

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
NUM_NODES=${12}
K=2

echo "@@@ RUNNING GCLSTM_no_norm on $DATA @@@"
echo "^^^ Number of units: $NUM_UNITS; number of channels: $OUT_CHANNELS; Chebyshev filter size: $K ^^^"

ARGS=(
    DTDG.link_pred.memory_node.gclstm_no_norm
    --data="$DATA"
    --seed=$SEED
    --patience=50
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
## To stop visualization overhead time for big graphs, it is allowed only on small number of nodes.
if [ $NUM_NODES -le 120 ]; then
    EVAL_ARGS=(
        "${ARGS[@]}" 
        "--num-epochs-to-visualize=1" 
        "--eval-mode"
    )
else
    EVAL_ARGS=(
        "${ARGS[@]}" 
        "--num-epochs-to-visualize=0" 
        "--eval-mode"
    )
fi

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
./plot/2d/memory_node/all_in_one.sh
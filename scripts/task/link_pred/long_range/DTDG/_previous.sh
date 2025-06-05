#!/bin/bash
#SBATCH --job-name=DT_MN_Previous
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

DATA_LOC=data/
RUN_SCRIPT=T-GRAB.train.run
NODE_POS=circular_layout

# Load module, env
module load python/3.8
source $PWD/tgrab/bin/activate
cd ../

DATA="$1"
ROOT_LOAD_SAVE_DIR=$2
WANDB_ENTITY=${3}
echo "@@@ RUNNING Previous on $DATA @@@"

ARGS=(
    DTDG.link_pred.memory_node.previous \
    --data="$DATA" \
    --data-loc=$DATA_LOC \
    --node-pos=$NODE_POS \
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project="T-GRAB" \
    --num-epochs-to-visualize=1 \
    --eval-mode
)

echo -e "\n\n %% START EVALUATION... %%"
python -m $RUN_SCRIPT "${ARGS[@]}"
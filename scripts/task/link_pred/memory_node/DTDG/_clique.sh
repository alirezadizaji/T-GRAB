#!/bin/bash
#SBATCH --job-name=DT_MN_Clique
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

DATA_LOC=lab/TSA/data/
RUN_SCRIPT=TSA.train.run
NODE_POS=circular_layout

# Load module, env
module load python/3.8
source $HOME/envs/tsa/bin/activate
cd $HOME/lab

DATA="$1"
ROOT_LOAD_SAVE_DIR=$2
WANDB_ENTITY=${3}
echo "@@@ RUNNING Clique on $DATA @@@"

ARGS=(
    DTDG.link_pred.memory_node.clique \
    --data="$DATA" \
    --data-loc=$DATA_LOC \
    --node-pos=$NODE_POS \
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project="TSA" \
    --num-epochs-to-visualize=1 \
    --eval-mode
)

echo -e "\n\n %% START EVALUATION... %%"
python -m $RUN_SCRIPT "${ARGS[@]}"
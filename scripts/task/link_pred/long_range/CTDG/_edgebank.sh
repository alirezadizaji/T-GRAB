#!/bin/bash
#SBATCH --job-name=CT_MN_EDGEBANK
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=long-cpu

DATA_LOC=lab/TSA/data/
RUN_SCRIPT=TSA.train.run_edgebank
NODE_POS=circular_layout

# Load module, env
module load python/3.8
source $PWD/tgrab/bin/activate
cd ../

# Edge bank scripts
DATA=$1
ROOT_LOAD_SAVE_DIR=$2
VAL_FIRST_METRIC="$3"
WANDB_ENTITY=${4}
MEM_MODE=unlimited
echo "^^^ RUNNING EDGEBANK on $DATA; memory mode: $MEM_MODE ^^^"
python -m $RUN_SCRIPT CTDG.link_pred.memory_node.edgebank \
    --mem_mode=$MEM_MODE \
    --data="$DATA" \
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR \
    --data-loc=$DATA_LOC \
    --node-pos=$NODE_POS \
    --val-first-metric="$VAL_FIRST_METRIC"
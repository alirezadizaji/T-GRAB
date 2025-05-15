#!/bin/bash
#SBATCH --job-name=CT_Pe_EDGEBANK
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=long-cpu

DATA_LOC=lab/TSA/data/
RUN_SCRIPT=TSA.train.run_edgebank
NODE_POS=circular_layout

# Load module, env
module load python/3.8
source $HOME/envs/tsa/bin/activate
cd $HOME/lab

# Edge bank scripts
DATA=$1
ROOT_LOAD_SAVE_DIR=$2
VAL_FIRST_METRIC="$3"
MEM_MODE=unlimited
WANDB_ENTITY=${4}
echo "^^^ RUNNING EDGEBANK on $DATA; memory mode: $MEM_MODE ^^^"
python -m $RUN_SCRIPT CTDG.link_pred.periodicity.edgebank \
    --mem_mode=$MEM_MODE \
    --data="$DATA" \
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR \
    --data-loc=$DATA_LOC \
    --node-pos=$NODE_POS \
    --val-first-metric="$VAL_FIRST_METRIC" \
    # --visualize \

# # Draw the plots
# echo -e "\n\n %% DRAW PLOTS... %%"
# cd $HOME/lab/TSA/scripts/
# ./plot/2d/periodicity/all_in_one.sh
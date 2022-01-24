#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=train # single job name for the array
#SBATCH --time=60:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=train.out # standard output
#SBATCH --error=train.err # standard error
# in the previous two lines %A" is replaced by job

#large can't do 16
#small 128 bs is tolerable

cd ~/TFix
source env/bin/activate
python hf_transformers/tfix_training.py -mn t5-small -e 25 -bs 16 -d repo-based

#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=train # single job name for the array
#SBATCH --time=6:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=train.out # standard output
#SBATCH --error=train.err # standard error
# in the previous two lines %A" is replaced by job

source env/bin/activate
python hf_transformers/tfix_training.py -mn t5-small -e 30 -bs 64 -d new

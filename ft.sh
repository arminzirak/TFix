#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=ft # single job name for the array
#SBATCH --time=1:20:00 # maximum walltime per job
#SBATCH --mem=20G # maximum 100M per job
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job


cd ~/TFix/
source env/bin/activate
python forward_translation.py
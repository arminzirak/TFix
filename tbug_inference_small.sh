#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=tb-ins # single job name for the array
#SBATCH --time=3:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job

cd ~/TFix/
source env/bin/activate
python tbug_inference.py
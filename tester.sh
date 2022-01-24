#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=test # single job name for the array
#SBATCH --time=0:40:00 # maximum walltime per job
#SBATCH --mem=15 # maximum 100M per job
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job

cd ~/TFix
source env/bin/activate
python hf_transformers/tfix_testing.py --load-model [] -bs 16 --model-name t5-small -d repo-based-included #-r /data/all/data/appium/appium

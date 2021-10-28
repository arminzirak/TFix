#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=test # single job name for the array
#SBATCH --time=2:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=test.out # standard output
#SBATCH --error=test.err # standard error
# in the previous two lines %A" is replaced by job

source env/bin/activate
python hf_transformers/tfix_testing.py --load-model /project/def-hemmati-ab/arminz/t5-small_global_new_27-10-2021_20-32-32/checkpoint-21200 -bs 64 --model-name t5-small -d new
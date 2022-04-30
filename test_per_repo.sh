#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=test # single job name for the array
#SBATCH --time=2:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=test.out # standard output
#SBATCH --error=test.err # standard error
# in the previous two lines %A" is replaced by job

source env/bin/activate
while IFS="," read -r ind repo sample train
do
  if [[ $train == 'False' ]]
  then
    echo $repo
    python hf_transformers/tfix_testing.py --load-model ./storage/checkpoint-37375/ -bs 32 --model-name t5-small -d repo-based-included -r $repo
  fi
done < <(tail -n +2 ./repos_3.csv)

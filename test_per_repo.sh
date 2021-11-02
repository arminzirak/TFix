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
while IFS="," read -r ind repo sample train
do
  if [[ $train == 'False' ]]
  then
    echo $repo
#    python hf_transformers/tfix_testing.py --load-model ./storage/checkpoint-40140 -bs 32 --model-name t5-small -d new -r $repo
    python hf_transformers/tfix_testing.py --load-model ./storage/data_and_models/models/t5small -bs 32 --model-name t5-small -d new -r $repo
  fi
done < <(tail -n +2 ./repos.csv)
# python hf_transformers/tfix_testing.py --load-model data_and_models/models/t5small -bs 32 --model-name t5-small -d new -r /data/all/data/appium/appium

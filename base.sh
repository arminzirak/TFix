#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=base # single job name for the array
#SBATCH --time=4:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=base.out # standard output
#SBATCH --error=base.err # standard error
# in the previous two lines %A" is replaced by job

source env/bin/activate

for repo in /data/all/data/appium/appium /data/all/data/girder/girder /data/all/data/oroinc/platform /data/all/data/svgdotjs/svg.js /data/all/data/wsick/Fayde /data/all/data/appium/appium /data/all/data/zloirock/core-js /data/all/data/Vincit/objection.js /data/all/data/request/request /data/all/data/qooxdoo/qooxdoo
do
    echo $repo
     python hf_transformers/tfix_testing.py --load-model ./storage/checkpoint-37375 -bs 64 --model-name t5-small -d repo-based-included -r $repo

done
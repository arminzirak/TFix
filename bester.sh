#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=best # single job name for the array
#SBATCH --time=6:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=best.out # standard output
#SBATCH --error=best.err # standard error
# in the previous two lines %A" is replaced by job

source env/bin/activate
for repo in /data/all/data/girder/girder /data/all/data/oroinc/platform /data/all/data/svgdotjs/svg.js /data/all/data/wsick/Fayde /data/all/data/appium/appium /data/all/data/zloirock/core-js /data/all/data/Vincit/objection.js /data/all/data/request/request /data/all/data/qooxdoo/qooxdoo
do
  for percent in 0.1 0.2 0.3
  do
    echo $repo $percent
    python best_params.py --percent $percent --repo $repo
  done
done
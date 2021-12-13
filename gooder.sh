#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=good # single job name for the array
#SBATCH --time=1:20:00 # maximum walltime per job
#SBATCH --mem=20G # maximum 100M per job
#SBATCH --cpus-per-task=1
#SBATCH --output=good.out # standard output
#SBATCH --error=good.err # standard error
# in the previous two lines %A" is replaced by job

#/data/all/data/appium/appium /data/all/data/girder/girder /data/all/data/oroinc/platform
source env/bin/activate
for repo in /data/all/data/svgdotjs/svg.js /data/all/data/wsick/Fayde /data/all/data/zloirock/core-js /data/all/data/Vincit/objection.js /data/all/data/request/request /data/all/data/qooxdoo/qooxdoo
do
  for percent in 0.1 0.3 0.7 1.0
  do
    echo $repo $percent
    python good_params.py --percent $percent --repo $repo
  done
done
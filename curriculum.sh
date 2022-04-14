#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=cl # single job name for the array
#SBATCH --time=5:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=cl.out # standard output
#SBATCH --error=cl.err # standard error
# in the previous two lines %A" is replaced by job

cd ~/TFix/
source env/bin/activate
for repo in /data/all/data/girder/girder /data/all/data/oroinc/platform /data/all/data/svgdotjs/svg.js /data/all/data/wsick/Fayde /data/all/data/appium/appium /data/all/data/zloirock/core-js /data/all/data/Vincit/objection.js /data/all/data/request/request /data/all/data/qooxdoo/qooxdoo
do
  echo $repo
  python curriculum.py -r $repo
done
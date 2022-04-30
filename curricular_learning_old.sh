#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=cl # single job name for the array
#SBATCH --time=20:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=cl.out # standard output
#SBATCH --error=cl.err # standard error
# in the previous two lines %A" is replaced by job

source env/bin/activate
#/data/all/data/appium/appium
for repo in /data/all/data/girder/girder /data/all/data/oroinc/platform /data/all/data/svgdotjs/svg.js /data/all/data/wsick/Fayde /data/all/data/appium/appium /data/all/data/zloirock/core-js /data/all/data/Vincit/objection.js /data/all/data/request/request /data/all/data/qooxdoo/qooxdoo
do
  for repo_percent in 0.3 0.6 1.0
  do
    for append in 50 450 800 #$(seq 850 400 3000)
    do
      echo $repo $repo_percent $append
      python add_bunch_of_data.py -a $append -r $repo -rp $repo_percent
    done
  done
done
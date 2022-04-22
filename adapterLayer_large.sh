#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=al_l # single job name for the array
#SBATCH --time=6:20:00 # maximum walltime per job
#SBATCH --mem=25G # maximum 100M per job
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job


cd ~/TFix/
source env/bin/activate
for repo in /data/all/data/qooxdoo/qooxdoo /data/all/data/elastic/kibana /data/all/data/emberjs/ember.js /data/all/data/zloirock/core-js /data/all/data/Encapsule-Annex/onm /data/all/data/sequelize/sequelize /data/all/data/dcos/dcos-ui /data/all/data/LivelyKernel/LivelyKernel /data/all/data/svgdotjs/svg.js /data/all/data/foam-framework/foam
do
  for percent in 1.0
  do
    echo $repo $percent
    python adapter_layer_tuner_large.py --percent $percent --repo $repo
  done
done
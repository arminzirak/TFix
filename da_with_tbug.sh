#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=tb-train # single job name for the array
#SBATCH --time=1:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job

cd ~/TFix/
source env/bin/activate

for repo in /data/all/data/qooxdoo/qooxdoo /data/all/data/elastic/kibana /data/all/data/emberjs/ember.js /data/all/data/zloirock/core-js /data/all/data/Encapsule-Annex/onm /data/all/data/sequelize/sequelize /data/all/data/dcos/dcos-ui /data/all/data/LivelyKernel/LivelyKernel /data/all/data/svgdotjs/svg.js /data/all/data/foam-framework/foam
do
  echo $repo
  python da_with_tbug.py -r $repo
done
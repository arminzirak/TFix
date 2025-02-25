#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=test-smiA # single job name for the array
#SBATCH --time=3:00:00 # maximum walltime per job
#SBATCH --mem=15G # maximum 100M per job
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job

cd ~/TFix
source env/bin/activate
for repo in /data/all/data/qooxdoo/qooxdoo /data/all/data/elastic/kibana /data/all/data/emberjs/ember.js /data/all/data/zloirock/core-js /data/all/data/Encapsule-Annex/onm /data/all/data/sequelize/sequelize /data/all/data/dcos/dcos-ui /data/all/data/LivelyKernel/LivelyKernel /data/all/data/svgdotjs/svg.js /data/all/data/foam-framework/foam
do
  echo $repo
  python hf_transformers/tfix_testing.py --load-model t5-small_repo-based-included_21-01-2022_10-30-33/checkpoint-17370 -bs 128 --model-name t5-small -d repo-based-included -r $repo
done
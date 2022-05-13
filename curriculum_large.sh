#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=cll # single job name for the array
#SBATCH --time=48:00:00 # maximum walltime per job
#SBATCH --mem=40G # maximum 100M per job
#SBATCH --cpus-per-task=2
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job

cd ~/TFix/
source env/bin/activate
for mode in conf length_label length_input
do
  for repo in /data/all/data/qooxdoo/qooxdoo /data/all/data/zloirock/core-js /data/all/data/emberjs/ember.js /data/all/data/foam-framework/foam /data/all/data/elastic/kibana  /data/all/data/Encapsule-Annex/onm /data/all/data/sequelize/sequelize /data/all/data/dcos/dcos-ui /data/all/data/LivelyKernel/LivelyKernel /data/all/data/svgdotjs/svg.js
  do
    echo $repo $mode
    model_address="/scratch/arminz/tmp/currl_"$repo'_'$mode
    python curriculum_large.py -r $repo -m $mode -md $model_address
    python hf_transformers/tfix_testing.py --load-model $model_address -bs 16 --model-name t5-large -d repo-based-included -r $repo
    python hf_transformers/tfix_testing.py --load-model $model_address -bs 16 --model-name t5-large -d source-test
  done
done
#!/bin/bash

#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=ftl # single job name for the array
#SBATCH --time=5:20:00 # maximum walltime per job
#SBATCH --mem=30G # maximum 100M per job
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job


cd ~/TFix/
source env/bin/activate
score_threshold=0
fw_epochs=1
for repo in /data/all/data/qooxdoo/qooxdoo /data/all/data/elastic/kibana /data/all/data/emberjs/ember.js /data/all/data/zloirock/core-js /data/all/data/Encapsule-Annex/onm /data/all/data/sequelize/sequelize /data/all/data/dcos/dcos-ui /data/all/data/LivelyKernel/LivelyKernel /data/all/data/svgdotjs/svg.js /data/all/data/foam-framework/foam
do
  echo $repo
  model_address="/scratch/arminz/tmp/ftl2_"$repo'_'$score_threshold'_'$fw_epochs
  python forward_translation_lg.py -r $repo --fw-epochs $fw_epochs --score-threshold $score_threshold -md $model_address
  echo $model_address
  python hf_transformers/tfix_testing.py --load-model $model_address -bs 16 --model-name t5-large -d repo-based-included -r $repo
  python hf_transformers/tfix_testing.py --load-model $model_address -bs 16 --model-name t5-large -d source-test
done

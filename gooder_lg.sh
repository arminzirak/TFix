#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=gooder_lg # single job name for the array
#SBATCH --time=12:00:00 # maximum walltime per job
#SBATCH --mem=30G # maximum 100M per job
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job

cd ~/TFix/
source env/bin/activate
for repo in /data/all/data/qooxdoo/qooxdoo /data/all/data/zloirock/core-js /data/all/data/emberjs/ember.js /data/all/data/foam-framework/foam /data/all/data/elastic/kibana  /data/all/data/Encapsule-Annex/onm /data/all/data/sequelize/sequelize /data/all/data/dcos/dcos-ui /data/all/data/LivelyKernel/LivelyKernel /data/all/data/svgdotjs/svg.js
do
  for percent in 1.0
  do
    echo $repo $percent
    python good_params_lg.py --percent $percent --repo $repo
    python hf_transformers/tfix_testing.py --load-model "/scratch/arminz/tmp/finetuned/$repo" -bs 8 --model-name t5-large -d repo-based-included -r $repo
    python hf_transformers/tfix_testing.py --load-model "/scratch/arminz/tmp/finetuned/$repo" -bs 8 --model-name t5-large -d source-test
  done
done

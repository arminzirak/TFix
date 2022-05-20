#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=testSA # single job name for the array
#SBATCH --time=0:40:00 # maximum walltime per job
#SBATCH --mem=25G # maximum 100M per job
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out # standard output
#SBATCH --error=%x.err # standard error
# in the previous two lines %A" is replaced by job

cd ~/TFix
source env/bin/activate
python hf_transformers/tfix_testing.py --load-model ~/scratch/training/t5-small_repo-based_21-01-2022_10-29-42/checkpoint-16440/ -bs 16 --model-name t5-small -d repo-based-included #-r /data/all/data/appium/appium

echo "finished general testing"
for repo in /data/all/data/qooxdoo/qooxdoo /data/all/data/zloirock/core-js /data/all/data/emberjs/ember.js /data/all/data/foam-framework/foam /data/all/data/elastic/kibana  /data/all/data/Encapsule-Annex/onm /data/all/data/sequelize/sequelize /data/all/data/dcos/dcos-ui /data/all/data/LivelyKernel/LivelyKernel /data/all/data/svgdotjs/svg.js
do
  python hf_transformers/tfix_testing.py --load-model ~/scratch/training/t5-small_repo-based_21-01-2022_10-29-42/checkpoint-16440/ -bs 16 --model-name t5-small -d repo-based-included -r $repo
done

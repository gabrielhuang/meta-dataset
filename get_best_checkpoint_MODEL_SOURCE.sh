#!/bin/bash
export MODEL=$1
export SOURCE=$2
do
  export EXPNAME=${MODEL}_${SOURCE}
  python2 -m meta_dataset.analysis.select_best_model \
    --all_experiments_root=$EXPROOT \
    --experiment_dir_basenames='' \
    --restrict_to_variants=${EXPNAME} \
    --description=best_${EXPNAME}
done

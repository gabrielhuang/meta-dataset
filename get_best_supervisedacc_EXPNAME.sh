#!/bin/bash
export EXPNAME=$1
python2 -m meta_dataset.analysis.select_best_model \
--all_experiments_root=$EXPROOT \
--experiment_dir_basenames='' \
--restrict_to_variants=${EXPNAME} \
--description=best_${EXPNAME}

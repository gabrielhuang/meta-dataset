#!/bin/bash
ulimit -n 100000 # increase open file limit
export SOURCE=imagenet  # all/imagenet
export CENTERLOSS=$2  # float, 1 is a decent default
export SINKHORNREG=${3:-10}
#for MODEL in baselinefinetune prototypical matching maml maml_init_with_proto
for MODEL in centroid
do
  export CONFIGNAME=${MODEL}_${SOURCE}
  export EXPNAME=${MODEL}_${SOURCE}_${CENTERLOSS}
  python2 -m meta_dataset.train \
    --records_root_dir=$RECORDS \
    --train_checkpoint_dir=${EXPROOT}/checkpoints/${EXPNAME} \
    --summary_dir=${EXPROOT}/summaries/${EXPNAME} \
    --gin_config=meta_dataset/learn/gin/best/${CONFIGNAME}.gin \
    --gin_bindings="LearnerConfig.experiment_name='$EXPNAME'" \
    --gin_bindings="LearnConfig.num_eval_episodes=600" \
    --gin_bindings="LearnConfig.num_eval_other_metrics=60" \
    --gin_bindings="LearnConfig.log_every=10" \
    --gin_bindings="DataConfig.shuffle_buffer_size=300" \
    --gin_bindings="CentroidNetworkLearner.center_loss=$CENTERLOSS" \
    --gin_bindings="CentroidNetworkLearner.sinkhorn_regularization=${SINKHORNREG}" 
done

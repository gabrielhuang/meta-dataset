#!/bin/bash
ulimit -n 100000 # increase open file limit
export SOURCE=all  # all/imagenet
#for MODEL in baselinefinetune prototypical matching maml maml_init_with_proto
for MODEL in prototypical
do
  export EXPNAME=${MODEL}_${SOURCE}
  python2 -m meta_dataset.train \
    --records_root_dir=$RECORDS \
    --train_checkpoint_dir=${EXPROOT}/checkpoints/${EXPNAME} \
    --summary_dir=${EXPROOT}/summaries/${EXPNAME} \
    --gin_config=meta_dataset/learn/gin/best/${EXPNAME}.gin \
    --gin_bindings="LearnerConfig.experiment_name='$EXPNAME'" \
    --gin_bindings="LearnConfig.num_eval_episodes=1000" \
    --gin_bindings="LearnConfig.log_every=10" \
    --gin_bindings="DataConfig.shuffle_buffer_size=300"  \
    --gin_bindings="LearnerConfig.pretrained_checkpoint='${EXPROOT}/checkpoints/baseline_imagenet_resnet/model_37000.ckpt'"
done

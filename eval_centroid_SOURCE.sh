#!/bin/bash
trap "exit" INT
export SOURCE=$1
for MODEL in centroid
do
  export EXPNAME=${MODEL}_${SOURCE}
  # set BESTNUM to the "best_update_num" field in the corresponding best_....txt
  export BESTNUM=$(grep best_update_num ${EXPROOT}/best_${EXPNAME}.txt | awk '{print $2;}')
  for DATASET in ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco
  do
    python2 -m meta_dataset.train \
      --is_training=False \
      --records_root_dir=$RECORDS \
      --summary_dir=${EXPROOT}/summaries/${EXPNAME}_eval_$DATASET \
      --gin_config=meta_dataset/learn/gin/best/${EXPNAME}.gin \
      --gin_bindings="LearnerConfig.experiment_name='${EXPNAME}'" \
      --gin_bindings="LearnerConfig.pretrained_checkpoint=''" \
      --gin_bindings="LearnerConfig.checkpoint_for_eval='${EXPROOT}/checkpoints/${EXPNAME}/model_${BESTNUM}.ckpt'" \
      --gin_bindings="benchmark.eval_datasets='$DATASET'" \
      --gin_bindings="LearnConfig.num_eval_episodes=600"
  done
done

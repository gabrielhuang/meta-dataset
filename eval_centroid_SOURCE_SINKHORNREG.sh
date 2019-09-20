#!/bin/bash
trap "exit" INT
export SOURCE=$1
export SINKHORNREG=$2
for MODEL in centroid
do
  export trainEXPNAME=${MODEL}_${SOURCE}
  export evalEXPNAME="${MODEL}_${SOURCE}_sinkhornreg_$SINKHORNREG"
  # set BESTNUM to the "best_update_num" field in the corresponding best_....txt
  export BESTNUM=$(grep best_update_num ${EXPROOT}/best_${EXPNAME}.txt | awk '{print $2;}')
  for DATASET in ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco
  do
    python2 -m meta_dataset.train \
      --is_training=False \
      --records_root_dir=$RECORDS \
      --summary_dir=${EXPROOT}/summaries/${evalEXPNAME}_eval_$DATASET \
      --gin_config=meta_dataset/learn/gin/best/${trainEXPNAME}.gin \
      --gin_bindings="LearnerConfig.experiment_name='${evalEXPNAME}'" \
      --gin_bindings="LearnerConfig.pretrained_checkpoint=''" \
      --gin_bindings="LearnerConfig.checkpoint_for_eval='${EXPROOT}/checkpoints/${trainEXPNAME}/model_${BESTNUM}.ckpt'" \
      --gin_bindings="benchmark.eval_datasets='$DATASET'" \
      --gin_bindings="LearnConfig.num_eval_episodes=600" \
      --gin_bindings="CentroidNetworkLearner.sinkhorn_regularization=$2" \
  done
done

#!/bin/bash
trap "exit" INT
export trainEXPNAME=$1
export SINKHORNREG=$2
export GINCONFIG=${3:-$1} # ginconfig is samae as trainExpname unless otherwise specified
export SOURCE=imagenet
export PREFIX=$4
for MODEL in centroid
do
  export evalEXPNAME="${PREFIX}${trainEXPNAME}_${SOURCE}_sinkhornreg_$SINKHORNREG"
  # set BESTNUM to the "best_update_num" field in the corresponding best_....txt
  export BESTNUM=$(grep best_update_num ${EXPROOT}/best_${trainEXPNAME}.txt | awk '{print $2;}')
  for DATASET in ilsvrc_2012
  do
    python2 -m meta_dataset.train \
      --is_training=False \
      --records_root_dir=$RECORDS \
      --summary_dir=${EXPROOT}/summaries/${evalEXPNAME}_eval_$DATASET \
      --gin_config=meta_dataset/learn/gin/best/${GINCONFIG}.gin \
      --gin_bindings="LearnerConfig.experiment_name='${evalEXPNAME}'" \
      --gin_bindings="LearnerConfig.pretrained_checkpoint=''" \
      --gin_bindings="LearnerConfig.checkpoint_for_eval='${EXPROOT}/checkpoints/${trainEXPNAME}/model_${BESTNUM}.ckpt'" \
      --gin_bindings="benchmark.eval_datasets='$DATASET'" \
      --gin_bindings="LearnConfig.num_eval_episodes=600" \
      --gin_bindings="CentroidNetworkLearner.sinkhorn_regularization=${SINKHORNREG}" 
  done
done

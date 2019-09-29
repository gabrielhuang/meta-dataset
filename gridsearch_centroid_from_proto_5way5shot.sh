#!/bin/bash
trap "exit" INT
for sinkhornreg in 1 3 10 30 100 300 1000 3000
do
	echo "Evaluating SinkhornReg=${sinkhornreg}"
	./eval_centroid_trainEXPNAME_SINKHORNREG_optGINCONFIG_onlyimagenet.sh prototypical_imagenet ${sinkhornreg} centroid_imagenet_5way_5shot 5way5shot_centroidFromProto_
done

#!/bin/bash
for sinkhornreg in 1 3 10 30 100 300 1000 3000
do
	echo "Evaluating SinkhornReg=${sinkhornreg}"
	./eval_centroid_trainEXPNAME_SINKHORNREG_optGINCONFIG_onlyimagenet.sh centroid_imagenet_1 ${sinkhornreg} centroid_imagenet
done

#!/bin/bash
EXPNAME=$1
UPDATE=$2
FILE="$EXPROOT/best_${EXPNAME}.txt"
echo "Editing ${FILE}"
echo "best_variant: $EXPROOT" > $FILE
echo "best_valid_acc: 0" >> $FILE
echo "best_update_num: $UPDATE" >> $FILE

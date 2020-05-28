#!bin/bash
if [ -z "$1" ]
  then
    echo "Need input GPU number"
    echo "Usage: bash `basename "$0"` \$GPU"
    exit
fi
gpu=$1
CUDA_VISIBLE_DEVICES=$gpu python trainer_osda.py \
    --net vgg \
    --source_path ../dataset/AID.txt \
    --target_path ../dataset/UCM_OPDA.txt \
    --dataset AID \
    --model_path checkpoint/checkpoint_99 \
    --test

#!bin/bash
if [ -z "$1" ]
  then
    echo "Need input base model!"
    echo "Usage: bash `basename "$0"` \$GPU"
    exit
fi
gpu=$1
CUDA_VISIBLE_DEVICES=$gpu python trainer_osda_sourceonly.py \
    --net vgg \
    --source_path ../dataset/AID.txt \
    --target_path ../dataset/UCM_OPDA.txt \
    --dataset AID \
    --model_path checkpoint_so_corr/checkpoint_so_corr_99 \
    --test


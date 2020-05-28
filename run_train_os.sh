#!bin/bash
gpu=$1
CUDA_VISIBLE_DEVICES=$gpu python trainer_osda_sourceonly.py \
    --net vgg \
    --source_path ../dataset/AID_UNK.txt \
    --target_path ../dataset/UCM_OPDA.txt \
    --dataset AID \
    --save \
    --train


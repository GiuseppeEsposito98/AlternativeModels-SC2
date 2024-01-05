#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/gesposito/AlternativeModels\-SC2

conda activate sc2-benchmark-fsim


python script/task/object_detection.py \
        --config configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq6ch_fpn_from_faster_rcnn_resnet50_fpn.yaml \
        --device cpu\
        --log script/task/logs/log_faster_rcnn.log 
        

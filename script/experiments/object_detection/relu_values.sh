#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/g.esposito/AlternativeModels\-SC2

conda activate sc2-benchmark


python script/task/object_detection_ssd_to_delete.py \
        --config configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq6ch_fpn_from_faster_rcnn_resnet50_fpn.yaml \
        --device cuda\
        --log script/task/logs/log_test_.log \
        --writerlog script/experiment/writers_logs_test_ \
        -test_only
        
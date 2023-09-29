#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/g.esposito/AltntiveModels\-SC2

conda activate sc2-benchmark



python /home/g.esposito/AlterntiveModels\-SC2/script/task/object_detection_ssd.py \
        --config /home/g.esposito/AltntiveModels\-SC2/configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml \
        --device cuda\
        -test_only \
        --log /home/g.esposito/AlterntiveModels\-SC2/script/task/logs/logs.log\




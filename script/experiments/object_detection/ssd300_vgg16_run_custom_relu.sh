#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/gesposito/AlternativeModels\-SC2

conda activate sc2-benchmark


python script/task/object_detection_ssd.py \
        --config configs/coco2017/supervised_compression/ghnd-bq/ssd300_vgg16-bq1ch_from_ssd300_vgg16_hardcut_custom_relu.yaml \
        --device cuda\
        --log script/task/logs/log_custom_relu.log \
        --writerlog script/experiments/writers_logs_custom_relu \
        

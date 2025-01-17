#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/gesposito/AlternativeModels\-SC2

conda activate sc2-benchmark


python script/task/image_classification.py \
        --config configs/ilsvrc2012/supervised_compression/ghnd-bq/MobileNetV3Small-bq6ch_from_MobileNetV3Small_hardcut_wo_avgpool.yaml \
        --device cpu\
        --log script/task/logs/mobilenet_v3_small_hardcut_wo_avgpool.log \
        --writerlog /home/gesposito/AlternativeModels\-SC2/script/task/writers_logs_mobilenet_v3_small_hardcut_wo_avgpool \

#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/jd.guerrero/Documents/g.esposito/AlternativeModels\-SC2

conda activate sc2-benchmark


python script/task/image_classification.py \
        --config configs/cifar100/MobileNetV3Small-bq6ch_from_MobileNetV3Small_cov13.yaml \
        --device cuda\
        --log script/task/logs/mobilenet_v3_small_cov13.log \
        --writerlog /home/jd.guerrero/Documents/g.esposito/AlternativeModels\-SC2/script/task/writers_logs_mobilenet_v3_small_cov13 \

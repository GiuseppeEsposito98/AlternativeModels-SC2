#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/gesposito/AlternativeModels\-SC2

conda activate sc2-benchmark


python script/task/image_classification_to_delete.py \
        --config configs/cifar100/MobileNetV3Small-bq6ch_from_MobileNetV3Small_cov13.yaml  \
        --device cuda\
        --log script/task/logs/mobilenet_v3_small_hardcut_keep_relu.log \
        -test_only\
        -student_only


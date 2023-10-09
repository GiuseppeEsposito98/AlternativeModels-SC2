#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/g.esposito/AlternativeModels\-SC2

conda activate sc2-benchmark


python script/task/object_detection_ssd.py \
        --config configs/coco2017/supervised_compression/ghnd-bq/ssd300_vgg16-bq1ch_from_ssd300_vgg16_hardcut_layer_swap.yaml \
        --device cuda\
        --log script/task/logs/log_layer_swap.log \
        --writerlog /home/g.esposito/AlternativeModels\-SC2/script/task/writers_logs_layer_swap \
#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/g.esposito/AltntiveModels\-SC2

conda activate sc2-benchmark



python /home/g.esposito/AltntiveModels\-SC2/script/task/object_detection_ssd.py \
        --config /home/g.esposito/AltntiveModels\-SC2/configs/coco2017/supervised_compression/ghnd-bq/ssd300_mobilenetv3-bq1ch_from_ssd300_mobilenetv3.yaml \
        --device cuda\
        --log /home/g.esposito/AltntiveModels\-SC2/script/task/logs/logs.log\

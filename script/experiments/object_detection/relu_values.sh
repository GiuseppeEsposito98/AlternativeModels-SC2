#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/g.esposito/AlternativeModels\-SC2

conda activate sc2-benchmark


python script/task/object_detection_ssd_to_delete.py \
<<<<<<< HEAD
        --config configs/coco2017/supervised_compression/ghnd-bq/ssd300_vgg16-bq1ch_from_ssd300_vgg16_hardcut_relu_values.yaml \
        --device cuda\
        --log script/task/logs/log_test_.log \
        --writerlog script/experiment/writers_logs_test_ \
        -test_only
=======
        --config configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq6ch_fpn_from_faster_rcnn_resnet50_fpn.yaml \
        --device cuda\
        --log script/task/logs/log_test_.log \
        --writerlog script/experiment/writers_logs_test_ \
        -test_only
        
>>>>>>> 09a9cde63ae34960a9c4e16f9519740a0cf7a56e

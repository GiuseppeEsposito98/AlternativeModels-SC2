#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=wo_maxpool
#SBATCH --mail-type=ALL
#SBATCH --output=/home/gesposito/AlternativeModels-SC2/script/task/logs/ssd_wo_maxpool.out

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/gesposito/AlternativeModels\-SC2

conda activate sc2-benchmark


python script/task/object_detection_ssd.py \
        --config configs/coco2017/supervised_compression/ghnd-bq/ssd300_vgg16-bq1ch_from_ssd300_vgg16_hardcut_wo_maxpool.yaml \
        --device cuda\
        --log script/task/logs/log_wo_maxpool.log \
        --writerlog script/experiments/writers_logs_wo_maxpool \
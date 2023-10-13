#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=custom_relu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/gesposito/AlternativeModels-SC2/script/task/logs/ssd_custom_relu.out

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/gesposito/AlternativeModels\-SC2

conda activate sc2-benchmark

module load nvidia/cudasdk/11.6


python script/task/object_detection_ssd.py \
        --config /home/gesposito/AlternativeModels-SC2/configs/coco2017/supervised_compression/ssd300_vgg16-bq1ch_from_ssd300_vgg16_hardcut_custom_relu.yaml \
        --device cuda\
        --log script/task/logs/log_custom_relu.log \
        --writerlog script/experiments/writers_logs_custom_relu \
        

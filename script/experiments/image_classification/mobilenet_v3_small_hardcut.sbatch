#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=hardcut
#SBATCH --mail-type=ALL

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/gesposito/AlternativeModels\-SC2

conda activate sc2-benchmark

module load nvidia/cudasdk/11.6


python script/task/image_classification.py \
        --config configs/ilsvrc2012/supervised_compression/ghnd-bq/MobileNetV3Small-bq6ch_from_MobileNetV3Small_hardcut.yaml \
        --device cuda\
        --log script/task/logs/mobilenet_v3_small_hardcut.log \
        --writerlog /home/gesposito/AlternativeModels\-SC2/script/task/writers_logs_mobilenet_v3_small_hardcut \

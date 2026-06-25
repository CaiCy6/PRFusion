#!/bin/bash
# ==============================================================================
# PRFusion Few-shot Training & Testing Scripts
# ==============================================================================
DATA_DIR="/path/to/Havard-Medical-Image-Fusion-Datasets-main"

for RATIO in 0.1 0.3 0.5; do
    for EXP in CT-MRI PET-MRI SPECT-MRI; do
        python shotTrainTest.py --train_ratio $RATIO --model_name PRFusion --method PRFusion --exp_name $EXP --data_dir $DATA_DIR --epochs 30 --gpu 0
    done
done

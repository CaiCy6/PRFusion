#!/bin/bash
# ==============================================================================
# PRFusion Ablation Study Training Scripts
# ==============================================================================
DATA_DIR="/path/to/Havard-Medical-Image-Fusion-Datasets-main"

for EXP in CT-MRI PET-MRI SPECT-MRI; do
    # PRFusionA1
    python train.py --model_name PRFusionA1 --method PRFusionA1 --exp_name $EXP --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0

    # PRFusionA2
    python train.py --model_name PRFusionA2 --method PRFusionA2 --exp_name $EXP --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0

    # PRFusionA3
    python train.py --model_name PRFusionA3 --method PRFusionA3 --exp_name $EXP --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0

    # PRFusionA4
    python train.py --model_name PRFusionA4 --method PRFusionA4 --exp_name $EXP --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0
done

# -------- Testing --------
for EXP in CT-MRI PET-MRI SPECT-MRI; do
    python test.py --model_name PRFusion --method PRFusion --exp_name $EXP --data_dir $DATA_DIR --gpu 0
done

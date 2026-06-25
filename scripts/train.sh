#!/bin/bash
# ==============================================================================
# PRFusion Training Scripts
# ==============================================================================
DATA_DIR="/path/to/Havard-Medical-Image-Fusion-Datasets-main"

# -------- MACTFusion (Baseline) --------
# CT-MRI
python train.py --model_name MACTFusion --method MACTFusion --exp_name CT-MRI --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0

# PET-MRI
python train.py --model_name MACTFusion --method MACTFusion --exp_name PET-MRI --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0

# SPECT-MRI
python train.py --model_name MACTFusion --method MACTFusion --exp_name SPECT-MRI --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0

# -------- PRFusion (Ours) --------
# CT-MRI
python train.py --model_name PRFusion --method PRFusion --exp_name CT-MRI --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0

# PET-MRI
python train.py --model_name PRFusion --method PRFusion --exp_name PET-MRI --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0

# SPECT-MRI
python train.py --model_name PRFusion --method PRFusion --exp_name SPECT-MRI --data_dir $DATA_DIR --epochs 50 --batch_size 4 --gpu 0

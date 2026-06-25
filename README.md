<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.6+-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-97CA00?style=flat&logo=open-source-initiative&logoColor=white" alt="License">
</p>

<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)">
    <img src="Arch.png" alt="PRFusion" width="75%">
  </picture>
</h1>

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                              ██████╗ ██████╗ ███████╗                        ║
║                              ██╔══██╗██╔══██╗██╔════╝                        ║
║                              ██████╔╝██████╔╝█████╗                          ║
║                              ██╔═══╝ ██╔══██╗██╔══╝                          ║
║                              ██║     ██║  ██║██║                             ║
║                              ╚═╝     ╚═╝  ╚═╝╚═╝                             ║
║                                                                              ║
║              Progressive RWKV-based Fusion Network for                       ║
║              Multi-Modal Medical Image Fusion                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

<div align="center">

> 📄 Submitted to **Applied Soft Computing** (ASOC) — Under Review

</div>

---

## Overview

PRFusion is a deep learning framework for multi-modal medical image fusion, covering **CT-MRI**, **PET-MRI**, and **SPECT-MRI** tasks. It supports full training with 5-fold cross-validation as well as few-shot experiments.

---

## Quick Start

### Installation

```bash
git clone https://github.com/CaiCy6/PRFusion.git
cd PRFusion
pip install -r requirements.txt
```

> 🔧 The CUDA WKV kernel compiles automatically on first run (`nvcc` required).

### Dataset Preparation

Download the [Harvard Medical Image Fusion Dataset](https://www.med.harvard.edu/AANLIB/home.html) and organize as:

```
Havard-Medical-Image-Fusion-Datasets-main/
├── CT-MRI/
│   ├── CT/        (*.png)
│   └── MRI/       (*.png)
├── PET-MRI/
│   ├── PET/       (*.png)
│   └── MRI/       (*.png)
└── SPECT-MRI/
    ├── SPECT/     (*.png)
    └── MRI/       (*.png)
```

### Training

```bash
python train.py \
    --model_name PRFusion \
    --method PRFusion \
    --exp_name CT-MRI \
    --data_dir /path/to/Havard-Medical-Image-Fusion-Datasets-main \
    --epochs 50 --batch_size 4 --gpu 0
```

| Option | Choices |
|:-------|:--------|
| `--model_name` | `MACTFusion` \| `PRFusion` \| `PRFusionA1` ~ `PRFusionA4` |
| `--exp_name`   | `CT-MRI` \| `PET-MRI` \| `SPECT-MRI` |

### Testing

```bash
python test.py \
    --model_name PRFusion \
    --method PRFusion \
    --exp_name CT-MRI \
    --data_dir /path/to/dataset \
    --gpu 0
```

### Few-shot

```bash
python shotTrainTest.py \
    --train_ratio 0.1 \
    --model_name PRFusion \
    --method PRFusion \
    --exp_name CT-MRI \
    --data_dir /path/to/dataset \
    --epochs 30 --gpu 0
```

---

## Project Structure

```
PRFusion/
├── Arch.png
├── README.md
├── requirements.txt
├── train.py                     # 5-fold CV training
├── test.py                      # Evaluation & metrics
├── shotTrainTest.py             # Few-shot training + testing
├── dataset.py                   # Data loader
├── loss.py                      # Fusion loss functions
├── eval_metrics.py              # MI, SSIM, PSNR, VIF, Qabf, SCD
├── logger.py
├── Fusionnet.py                 # MACTFusion baseline
├── CrossMaxvit.py               # MaxViT cross-attention blocks
├── Maxvit.py                    # MaxViT blocks
├── Networks/
│   ├── FusionNet.py             # PRFusion (proposed)
│   ├── OursA1.py ~ OursA4.py    # Ablation variants
│   └── OursFusionNet/           # RWKV 2D modules & CUDA kernels
│       └── cuda/                # WKV forward/backward operators
└── scripts/
    ├── train.sh
    ├── train_ablation.sh
    └── train_fewshot.sh
```

---

## Key Arguments

| Argument | Default | Description |
|:---------|:-------:|:------------|
| `--model_name` | `MACTFusion` | Model architecture |
| `--exp_name` | `SPECT-MRI` | Dataset / fusion task |
| `--method` | `MACTFusion` | Save directory name |
| `--data_dir` | *required* | Path to dataset root |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `4` | Batch size per GPU |
| `--lr_start` | `0.001` | Initial learning rate |
| `--lr_decay_type` | `step` | `step` / `cosine` / `plateau` / `poly` |
| `--optimizer` | `adam` | `adam` / `adamw` / `sgd` |
| `--num_folds` | `5` | K-fold cross-validation |
| `--gpu` | `0` | GPU device ID |

---

## Citation

```bibtex
@article{PRFusion2025,
  title   = {PRFusion: Progressive RWKV-based Fusion Network
             for Multi-Modal Medical Image Fusion},
  journal = {Applied Soft Computing},
  note    = {Under review}
}
```

---

## License

MIT.

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║       Made with ❤️  for the Medical Image Community           ║
╚══════════════════════════════════════════════════════════════╝
```

</div>

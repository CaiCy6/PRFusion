<div align="center">

```
██████╗ ██████╗ ███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔══██╗██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║
██████╔╝██████╔╝█████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║
██╔═══╝ ██╔══██╗██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║
██║     ██║  ██║██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║
╚═╝     ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
</div>

<div align="center">

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          Progressive RWKV-based Fusion Network for                           ║
║          Multi-Modal Medical Image Fusion                                    ║
║                                                                              ║
║   [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()            ║
║   [![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)]()       ║
║   [![CUDA](https://img.shields.io/badge/CUDA-11.6+-76b900.svg)]()             ║
║   [![License](https://img.shields.io/badge/License-MIT-green.svg)]()          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

</div>

---

> 📄 This work has been submitted to **Applied Soft Computing (ASOC)**.

---

## Architecture

<div align="center">
  <img src="Arch.png" alt="PRFusion Architecture" width="85%">
</div>

---

## Overview

PRFusion is a multi-modal medical image fusion framework for **CT-MRI**, **PET-MRI**, and **SPECT-MRI** tasks. It supports standard training with 5-fold cross-validation and few-shot experiments.

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

> CUDA WKV kernels are compiled on first run (`nvcc` required).

### Dataset

```
Havard-Medical-Image-Fusion-Datasets-main/
├── CT-MRI/   {CT/*.png, MRI/*.png}
├── PET-MRI/  {PET/*.png, MRI/*.png}
└── SPECT-MRI/{SPECT/*.png, MRI/*.png}
```

### Train

```bash
python train.py \
    --model_name PRFusion \
    --method PRFusion \
    --exp_name CT-MRI \
    --data_dir /path/to/dataset \
    --epochs 50 --batch_size 4 --gpu 0
```

**Models:** `MACTFusion` | `PRFusion` | `PRFusionA1` ~ `PRFusionA4`

**Datasets:** `CT-MRI` | `PET-MRI` | `SPECT-MRI`

### Test

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
├── Arch.png                     # Architecture diagram
├── README.md
├── requirements.txt
├── train.py                     # Training with 5-fold CV
├── test.py                      # Evaluation & metrics
├── shotTrainTest.py             # Few-shot training + testing
├── dataset.py                   # Data loader
├── loss.py                      # Fusion loss
├── eval_metrics.py              # Evaluation metrics
├── logger.py                    # Logging
├── Fusionnet.py                 # MACTFusion model
├── CrossMaxvit.py               # Vision backbone
├── Maxvit.py                    # Vision backbone
├── Networks/
│   ├── FusionNet.py             # PRFusion (proposed)
│   ├── OursA1.py ~ OursA4.py    # Ablation variants
│   └── OursFusionNet/           # RWKV 2D blocks & CUDA kernels
└── scripts/
    ├── train.sh
    ├── train_ablation.sh
    └── train_fewshot.sh
```

---

## Key Arguments

| Argument | Default | Description |
|:---------|:-------:|:------------|
| `--model_name` | MACTFusion | Model selection |
| `--exp_name` | SPECT-MRI | Dataset / task |
| `--data_dir` | required | Dataset root path |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr_start` | 0.001 | Initial learning rate |
| `--num_folds` | 5 | Cross-validation folds |
| `--gpu` | 0 | GPU device ID |

---

## Citation

```bibtex
@article{PRFusion2025,
  title={PRFusion: Progressive RWKV-based Fusion Network for Multi-Modal Medical Image Fusion},
  journal={Applied Soft Computing},
  note={Under review}
}
```

---

## License

MIT License.

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║  Made with ❤️  for the Medical Image Analysis Community      ║
╚══════════════════════════════════════════════════════════════╝
```

</div>

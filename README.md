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

## Overview

PRFusion is a multi-modal medical image fusion framework supporting **CT-MRI**, **PET-MRI**, and **SPECT-MRI** fusion tasks. It operates in the YCrCb color space and supports both standard and few-shot training with 5-fold cross-validation.

---

## Project Structure

```
PRFusion/
├── README.md
├── requirements.txt
├── train.py                     # Training with 5-fold CV
├── test.py                      # Evaluation & metrics
├── shotTrainTest.py             # Few-shot training + testing
├── dataset.py                   # Data loader (K-Fold)
├── loss.py                      # Fusion loss
├── eval_metrics.py              # MI, SSIM, PSNR, VIF, Qabf, SCD
├── logger.py                    # Logging utilities
├── Fusionnet.py                 # MACTFusion model
├── CrossMaxvit.py / Maxvit.py   # Vision backbone components
├── Networks/
│   ├── FusionNet.py             # PRFusion (proposed)
│   ├── OursA1.py ~ OursA4.py    # Ablation variants
│   └── OursFusionNet/           # RWKV & CUDA kernels
└── scripts/
    ├── train.sh
    ├── train_ablation.sh
    └── train_fewshot.sh
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

> CUDA WKV kernels are compiled on first run (`nvcc` required).

### Dataset

Organize the [Harvard Medical Image Fusion Dataset](https://www.med.harvard.edu/AANLIB/home.html) as:

```
Havard-Medical-Image-Fusion-Datasets-main/
├── CT-MRI/
│   ├── CT/        # *.png
│   └── MRI/       # *.png
├── PET-MRI/
│   ├── PET/       # *.png
│   └── MRI/       # *.png
└── SPECT-MRI/
    ├── SPECT/     # *.png
    └── MRI/       # *.png
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

**Models:** `MACTFusion` | `PRFusion` | `PRFusionA1` ~ `PRFusionA4`

**Datasets:** `CT-MRI` | `PET-MRI` | `SPECT-MRI`

### Testing

```bash
python test.py \
    --model_name PRFusion \
    --method PRFusion \
    --exp_name CT-MRI \
    --data_dir /path/to/Havard-Medical-Image-Fusion-Datasets-main \
    --gpu 0
```

### Few-shot

```bash
python shotTrainTest.py \
    --train_ratio 0.1 \
    --model_name PRFusion \
    --method PRFusion \
    --exp_name CT-MRI \
    --data_dir /path/to/Havard-Medical-Image-Fusion-Datasets-main \
    --epochs 30 --gpu 0
```

---

## Evaluation Metrics

| Metric | Description |
|:------:|:------------|
| MI | Mutual Information |
| CC | Correlation Coefficient |
| PSNR | Peak Signal-to-Noise Ratio |
| SSIM | Structural Similarity |
| VIFF | Visual Information Fidelity |
| SCD | Sum of Correlation Differences |
| Qabf | Quality of Blended Images |

---

## Key Arguments

| Argument | Default | Description |
|:---------|:-------:|:------------|
| `--model_name` | MACTFusion | Model to use |
| `--exp_name` | SPECT-MRI | Dataset / task |
| `--data_dir` | (required) | Path to dataset root |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr_start` | 0.001 | Initial learning rate |
| `--num_folds` | 5 | K-fold cross-validation |
| `--gpu` | 0 | GPU device ID |

---

## Citation

```bibtex
@article{PRFusion2025,
  title={PRFusion: Progressive RWKV-based Fusion Network for Multi-Modal Medical Image Fusion},
  author={},
  journal={},
  year={2025},
  publisher={}
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

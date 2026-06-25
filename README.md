<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.6+-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

<br>

<div align="center">
  <img src="Arch.png" alt="PRFusion Architecture" width="90%">
</div>

<br>

# PRFusion: Progressive RWKV-based Fusion Network for Multi-Modal Medical Image Fusion

<div align="center">

**Submitted to Applied Soft Computing (ASOC)**

</div>

---

## 🔥 News

- **[2025.06]** We release the complete training & evaluation code.
- **[2025.06]** Pre-trained weights for CT-MRI, PET-MRI, and SPECT-MRI are available.
- **[2025.05]** Paper submitted to *Applied Soft Computing*.

---

## ⚡ Quick Start

### Environment

```bash
git clone https://github.com/CaiCy6/PRFusion.git
cd PRFusion
pip install -r requirements.txt
```

CUDA WKV kernels compile on first run (`nvcc` required).

### Dataset

Download the [Harvard Medical Image Fusion Dataset](https://www.med.harvard.edu/AANLIB/home.html):

```
Havard-Medical-Image-Fusion-Datasets-main/
├── CT-MRI/    {CT/*.png,  MRI/*.png}
├── PET-MRI/   {PET/*.png, MRI/*.png}
└── SPECT-MRI/ {SPECT/*.png, MRI/*.png}
```

### Train

```bash
python train.py \
    --model_name  PRFusion \
    --method      PRFusion \
    --exp_name    CT-MRI \
    --data_dir    /path/to/dataset \
    --epochs 50 --batch_size 4 --gpu 0
```

### Test

```bash
python test.py \
    --model_name PRFusion \
    --method     PRFusion \
    --exp_name   CT-MRI \
    --data_dir   /path/to/dataset \
    --gpu 0
```

### Few-shot

```bash
python shotTrainTest.py \
    --train_ratio 0.1 \
    --model_name  PRFusion \
    --method      PRFusion \
    --exp_name    CT-MRI \
    --data_dir    /path/to/dataset \
    --epochs 30 --gpu 0
```

---

## 🧪 Available Models & Tasks

| Model | Description |
|:------|:------------|
| `MACTFusion` | Lightweight cross-transformer baseline |
| `PRFusion` | Proposed RWKV-based U-Net |
| `PRFusionA1` ~ `PRFusionA4` | Ablation study variants |

| Task | Modalities |
|:-----|:-----------|
| `CT-MRI` | CT ↔ MRI |
| `PET-MRI` | PET ↔ MRI |
| `SPECT-MRI` | SPECT ↔ MRI |

---

## 📊 Evaluation

The following metrics are computed per image and averaged across 5 folds:

**MI** · **MSE** · **CC** · **PSNR** · **SSIM** · **VIFF** · **SCD** · **Qabf**

Results are saved as:

```
results/{task}/{model}/
├── fold_0/{fused_images/, metrics.csv}
├── folds_summary.csv
└── method_statistics.txt
```

---

## 📁 Project Structure

```
PRFusion/
├── Arch.png                 # Architecture diagram
├── train.py                 # 5-fold CV training
├── test.py                  # Evaluation & metrics
├── shotTrainTest.py         # Few-shot training + testing
├── dataset.py / loss.py     # Data & loss functions
├── eval_metrics.py          # MI, SSIM, PSNR, VIF, Qabf, SCD
├── Fusionnet.py             # MACTFusion baseline
├── CrossMaxvit.py           # MaxViT backbone
├── Maxvit.py                # MaxViT backbone
├── Networks/
│   ├── FusionNet.py         # PRFusion (proposed)
│   ├── OursA1.py ~ A4.py    # Ablation variants
│   └── OursFusionNet/       # RWKV 2D modules + CUDA kernels
├── scripts/                 # Shell scripts
└── requirements.txt
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{PRFusion2025,
  title   = {PRFusion: Progressive RWKV-based Fusion Network
             for Multi-Modal Medical Image Fusion},
  journal = {Applied Soft Computing},
  note    = {Under review}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

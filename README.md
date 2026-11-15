<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.6+-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.asoc.2026.116704-blue)](https://doi.org/10.1016/j.asoc.2026.116704)

</div>

<br>

<div align="center">
  <img src="Arch.png" alt="PRFusion Architecture" width="90%">
</div>

<br>

# PRFusion: A Progressive Receptance Weighted Key Value Network for Multimodal Medical Image Fusion

> **医学图像融合模型** —— 基于 RWKV（Receptance Weighted Key Value）架构的渐进式跨模态融合网络，支持 CT-MRI / PET-MRI / SPECT-MRI 多模态融合。
> 论文已发表于 *Applied Soft Computing*（Vol. 204, Part E, Article 116704, 2027）。

<div align="center">

**Published in Applied Soft Computing (ASOC)**
Volume 204, Part E, Article 116704 · January 2027

[![ScienceDirect](https://img.shields.io/badge/ScienceDirect-10.1016%2Fj.asoc.2026.116704-red?style=flat-square)](https://www.sciencedirect.com/science/article/pii/S1568494626021526)

</div>

---

## 🔥 News

- **[2026.11]** Paper accepted & published in *Applied Soft Computing* — **Volume 204, Part E, Article 116704** (January 2027).
- **[2025.06]** We release the complete training & evaluation code.
- **[2025.06]** Pre-trained weights for CT-MRI, PET-MRI, and SPECT-MRI are available.
- **[2025.05]** Paper submitted to *Applied Soft Computing*.

---

## 📄 Paper & How to Get It

**Official title:** *A progressive receptance weighted key value network for multimodal medical image fusion*
**Authors:** Yang Zhou, Yifan Wang, Hongyu Cai

| Resource | Link |
|:--|:--|
| DOI | https://doi.org/10.1016/j.asoc.2026.116704 |
| ScienceDirect (article page) | https://www.sciencedirect.com/science/article/pii/S1568494626021526 |
| ScienceDirect (abstract) | https://www.sciencedirect.com/science/article/abs/pii/S1568494626021526 |
| Volume 204, Part E — Table of contents | https://www.sciencedirect.com/journal/applied-soft-computing/vol/204/part/PE |
| **Code (this repository)** | https://github.com/CaiCy6/PRFusion |

**Full-text access** — the article is published in a subscription journal (not Open Access), so the PDF is obtained via one of:

1. **Institutional access** — sign in with your organisation on the [ScienceDirect article page](https://www.sciencedirect.com/science/article/pii/S1568494626021526) (*"Get full-text access by signing in with your organisation"*). If your university / hospital / library subscribes to Elsevier, this works directly.
2. **Elsevier TDM** — machine-readable full text via the [Elsevier API](https://api.elsevier.com/content/article/PII:S1568494626021526) (requires an API key; also subject to the publisher's TDM licence).
3. **Elsevier / ScienceDirect article purchase** — one-off access via the publisher's paywall page on the article page.
4. **Abstract + highlights (free)** — the abstract, highlights and section snippets are openly readable on the abstract page; the full PDF requires any of the above.

> The **code, pre-trained weights and evaluation scripts in this repository are free and MIT-licensed** — no subscription needed to reproduce the results.

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

## 🧪 Supported Tasks

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
├── Networks/
│   ├── FusionNet.py         # PRFusion
│   └── OursFusionNet/       # RWKV 2D modules + CUDA kernels
├── scripts/                 # Shell scripts
└── requirements.txt
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2026prfusion,
  title   = {A progressive receptance weighted key value network for multimodal medical image fusion},
  author  = {Zhou, Yang and Wang, Yifan and Cai, Hongyu},
  journal = {Applied Soft Computing},
  year    = {2027},
  volume  = {204},
  pages   = {116704},
  doi     = {10.1016/j.asoc.2026.116704},
  url     = {https://doi.org/10.1016/j.asoc.2026.116704}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).
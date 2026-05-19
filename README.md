# Sensitivity Based Augmentation (Detectron2 Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-orange.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-2406.01425-b31b1b.svg)](https://arxiv.org/html/2406.01425v5)
[![Framework](https://img.shields.io/badge/Framework-Detectron2-FF6B6B.svg)](https://github.com/facebookresearch/detectron2)

**Status: Work in Progress** 🚧

This project ports **Sensitivity-Informed Augmentation (SensAug)** from MMSegmentation to **Detectron2**, an instance segmentation framework. The core logic and principles of SensAug remain unchanged. We're adapting the framework to support instance-level segmentation tasks.

### Citation

This project is based on the SensAug framework by Zheng et al. If you use this work, please cite:

**Paper:** [Sensitivity-Informed Augmentation for Robust Segmentation](https://arxiv.org/html/2406.01425v5)

**Original Repository:** [https://github.com/laurayuzheng/SensAug](https://github.com/laurayuzheng/SensAug)

## Project Overview

SensAug is a data augmentation technique that improves robustness in segmentation models by focusing augmentation efforts on regions the model is sensitive to. This version leverages Detectron2's capabilities for instance segmentation workflows.

## Getting Started

### Environment Setup (Conda)

First, create a conda environment with Python version 3.10:

```bash
conda create --name sensaug-d2 python=3.10
```

Then, install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```



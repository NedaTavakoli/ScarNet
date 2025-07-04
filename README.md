# ScarNet: Automated Myocardial Scar Quantification from LGE Cardiac MRI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
ScarNet is a deep learning model for automated myocardial scar segmentation in Late Gadolinium Enhancement (LGE) Cardiac MRI. It combines MedSAM's Vision Transformer encoder with a U-Net decoder for precise scar boundary detection.

## Architecture
![ScarNet Architecture](figures/Fig1.png)

![Detailed Architecture](figures/Fig2.png)

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU

### Setup
```bash
git clone https://github.com/NedaTavakoli/ScarNet.git
cd ScarNet
pip install -r requirements.txt
```

### Data Structure
```
data/
├── training/
│   ├── images/
│   └── masks/
├── validation/
│   ├── images/
│   └── masks/
└── testing/
    ├── images/
    └── masks/
```

## Usage

### Training
```bash
python train.py --config config.yaml
```

### Inference
```bash
python inference.py --input_dir /path/to/images --output_dir /path/to/results
```

### Configuration Example
```yaml
model:
  architecture: "ScarNet"
  
training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 100
  
loss:
  lambda1: 0.5  # Focal Tversky Loss
  lambda2: 0.4  # DICE Loss  
  lambda3: 0.1  # Cross-Entropy Loss
```

# ScarNet: A Novel Foundation Model for Automated Myocardial Scar Quantification

## Overview
ScarNet is a deep learning model specifically designed for automated myocardial scar segmentation and quantification in Late Gadolinium Enhancement (LGE) Cardiac MRI. By leveraging a hybrid architecture that integrates MedSAM's Vision Transformer (ViT)-based encoder with a U-Net decoder, ScarNet achieves state-of-the-art performance in scar segmentation and quantification, significantly surpassing existing methods like MedSAM and nnU-Net.

## Features
1. **Hybrid Architecture**:
   - Combines the global context awareness of transformers with the precise localization capabilities of CNNs.
   - Incorporates specialized attention mechanisms, including SE layers and ScarAttention blocks.

2. **Innovative Loss Function**:
   - Combines Dice, Focal, and Cross-Entropy losses to address class imbalance and improve segmentation accuracy.

3. **Robust Performance**:
   - Demonstrates high Dice scores (median = 0.912) for scar segmentation, outperforming existing models.
   - Robust against noise perturbations, ensuring reliability in varying imaging conditions.

4. **Efficient Inference Pipeline**:
   - Designed for seamless integration into clinical workflows.

## Key Results
- ScarNet achieves a median scar Dice score of 0.912, compared to 0.046 for MedSAM and 0.638 for nnU-Net.
- Demonstrates exceptional robustness across different image qualities and noise conditions.

## Figures
### Model Architecture
**Figure 1:** Hybrid architecture combining MedSAM's ViT-based encoder with U-Net's multi-scale decoder. This figure highlights ScarNet's integration of attention mechanisms for precise scar segmentation.

![Figure 1](ScarNet/figures/Figure1.png)

**Figure 2:** Detailed encoder-decoder overview, showcasing attention blocks and SE layers that enhance feature extraction and refine scar segmentation.

![Figure 2](ScarNet/figures/Figure2.png)

### Comparative Analysis
**Figure 3:** Comparative visualization of ScarNet, MedSAM, and nnU-Net segmentations. ScarNet demonstrates superior segmentation accuracy across different cardiac regions.

![Figure 3](ScarNet/figures/Figure3.png)

**Figure 4:** Performance analysis across varying training data sizes. ScarNet consistently outperforms MedSAM and nnU-Net in both myocardium and scar segmentation.

![Figure 4](ScarNet/figures/Figure4.png)

**Figure 5:** Representative test cases demonstrating ScarNet's segmentation performance compared to MedSAM and nnU-Net.

![Figure 5](ScarNet/figures/Figure5.png)

**Figure 6:** Performance metrics for scar volume quantification, including Bland-Altman plots and correlation analyses. ScarNet achieves near-perfect correlation with manual segmentation.

![Figure 6](ScarNet/figures/Figure6.png)

## Methodology
### Dataset
- Retrospective analysis using de-identified LGE images from 736 patients with ischemic cardiomyopathy.
- Training set: 552 patients; Testing set: 184 patients.

### Training Pipeline
1. Image preprocessing and augmentation.
2. Parallel processing through MedSAM and U-Net pathways.
3. Adaptive feature fusion and output generation.

### Loss Function
The multi-component loss function is defined as:
\[ \mathcal{L}_{ScarNet} = \lambda_1 \text{FTL} + \lambda_2 \text{DL} + \lambda_3 \text{CE} \]
- FTL: Focal Tversky Loss
- DL: Dice Loss
- CE: Cross-Entropy Loss

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NedaTavakoli/ScarNet.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model:
   ```bash
   python train_scar_net.py
   ```

## Citation
If you use ScarNet in your research, please cite the following paper:
```bibtex
@article{tavakoli2024scarnet,
  title={ScarNet: A Novel Foundation Model for Automated Myocardial Scar Quantification from LGE in Cardiac MRI},
  author={Neda Tavakoli et al.},
  journal={arXiv},
  year={2024}
}
```

## License
This project is licensed under the MIT License.

## Contact
For further information, please contact:
Neda Tavakoli  
Email: [neda.tavakoli@northwestern.edu](mailto:neda.tavakoli@northwestern.edu)


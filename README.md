# Pixel-Level Uncertainty Landslide Mapping

Official implementation of **Improving Deep Learning Landslide Mapping Performance Based on Pixel-Level Uncertainty Analysis**, built on MMSegmentation.

## Environment
- Python 3.9
- PyTorch 2.1.0 + CUDA 11.8
- MMCV 2.1.0
- MMEngine 0.10.7
- MMSegmentation 1.2.2

## Repository Structure
- `configs/`: Baseline and improved model config files
- `models/`: Custom decoder implementation
- `uncertainty/`: Pixel-level uncertainty analysis and MC Dropout inference code
- `utils/`: Data preprocessing and visualization scripts

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare landslide dataset and modify data paths in config files
3. Train / infer with MMSegmentation tools
4. Run scripts in `uncertainty/` for uncertainty analysis
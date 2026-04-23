# DFMGNet

Official PyTorch implementation of **DFMGNet: RGB-D Camouflaged Object Detection With Mamba Fusion and Dynamic Frequency-aware Refinement**.

## Highlights

- **MGAF**: a Mamba-Guided Attention Fusion module for adaptive RGB-D cross-modal interaction.
- **FAME**: a Frequency-Aware Multi-scale Enhancement module for dynamic high-frequency refinement.
- **Iterative decoder**: progressively restores object structures and improves boundary quality.
- Supports **RGB-D COD** benchmarks including **CAMO**, **CHAMELEON**, **COD10K**, and **NC4K**.

## Framework Overview

DFMGNet follows a dual-branch RGB-D pipeline. RGB images and depth maps are first encoded by two PVT backbones, then fused by MGAF, refined by FAME in the frequency domain, and finally decoded to generate the camouflaged object prediction and auxiliary edge prediction.

## Repository Structure

```text
DFMGNet/
├── models/
│   ├── DFMG.py          # Main network definition
│   ├── pvtv2.py         # PVTv2 backbone
│   └── vmamba.py        # Mamba-related modules
├── tools/
│   ├── data.py          # Data loading and augmentation
│   ├── logger.py
│   ├── lr_scheduler.py
│   └── utils.py
├── pytorch_iou/
│   └── __init__.py      # IoU loss
├── train.py             # Training script
├── test.py              # Inference script
├── speed.py             # Params / FLOPs / FPS benchmarking
├── options.py           # Training options
├── LICENSE
└── README.md
```

## Requirements

Recommended environment:

- Python 3.8+
- PyTorch 1.12+ / 2.x
- CUDA-enabled GPU

Main dependencies:

```bash
torch
torchvision
numpy
opencv-python
Pillow
tqdm
tensorboardX
einops
timm
matplotlib
onnx
fvcore    # optional, for profiling
thop      # optional, for profiling
ptflops   # optional, for profiling
termcolor
```

You can install the common dependencies with:

```bash
pip install torch torchvision numpy opencv-python pillow tqdm tensorboardX einops timm matplotlib onnx termcolor
```

Optional profiling packages:

```bash
pip install fvcore thop ptflops
```

## Dataset Preparation

The code expects the following directory structure:

```text
Data/
└── COD/
    ├── train/
    │   ├── RGB/
    │   ├── depth/
    │   ├── GT/
    │   └── Edge/
    └── test/
        ├── CAMO/
        │   ├── RGB/
        │   ├── depth/
        │   ├── GT/
        │   └── Edge/
        ├── CHAMELEON/
        │   ├── RGB/
        │   ├── depth/
        │   └── GT/
        ├── COD10K/
        │   ├── RGB/
        │   ├── depth/
        │   └── GT/
        └── NC4K/
            ├── RGB/
            ├── depth/
            └── GT/
```

### Notes

- Training uses **RGB**, **depth**, **GT**, and **Edge** maps.
- Testing in `test.py` requires **RGB**, **depth**, and **GT** folders for each dataset.
- File names across RGB, depth, GT, and edge maps should match.
- The current implementation assumes grayscale depth maps and repeats them to 3 channels before feeding them into the model.

## Training

Before training, modify the dataset and checkpoint paths in `options.py` if needed.

Default important options:

- `--epoch`: number of epochs
- `--lr`: learning rate
- `--batchsize`: training batch size
- `--trainsize`: training image size
- `--load`: path to pre-trained PVT weights
- `--save_path`: directory to save checkpoints and logs

Example:

```bash
python train.py \
  --epoch 80 \
  --lr 1e-4 \
  --batchsize 7 \
  --trainsize 384 \
  --load ./pretrain/pvt_v2_b5.pth \
  --rgb_root ../Data/COD/train/RGB/ \
  --depth_root ../Data/COD/train/depth/ \
  --gt_root ../Data/COD/train/GT/ \
  --edge_root ../Data/COD/train/Edge/ \
  --test_rgb_root ../Data/COD/test/CAMO/RGB/ \
  --test_depth_root ../Data/COD/test/CAMO/depth/ \
  --test_gt_root ../Data/COD/test/CAMO/GT/ \
  --save_path ./checkpoints/ckpt/
```

## Inference

Place the trained model at:

```text
./checkpoints/ckpt/DFMG_epoch_best.pth
```

Then run:

```bash
python test.py --gpu_id 0 --test_path ../Data/COD/test/
```

Prediction maps will be saved to:

```text
./final/COD_result/
├── CAMO/
├── CHAMELEON/
├── COD10K/
└── NC4K/
```

Auxiliary edge predictions are saved under each dataset's `edge/` subfolder.

## Pretrained Model and Code Package

### Baidu Netdisk

- **Code / result package (`final.zip`)**  
  Link: [https://pan.baidu.com/s/1gJN_ac6Lnd0KwyqyoTuVLQ?pwd=2025](https://pan.baidu.com/s/1gJN_ac6Lnd0KwyqyoTuVLQ?pwd=2025)  
  Extraction code: `2025`

- **Pretrained model (`DFMG_epoch_best.pth`)**  
  Link: [https://pan.baidu.com/s/165KLBru6dYmhwBZBScvX2g?pwd=2025](https://pan.baidu.com/s/165KLBru6dYmhwBZBScvX2g?pwd=2025)  
  Extraction code: `2025`

### Placement

Download the pretrained checkpoint and put it in:

```text
./checkpoints/ckpt/DFMG_epoch_best.pth
```

If you use pre-trained PVT weights for training, place them in a path such as:

```text
./pretrain/pvt_v2_b5.pth
```

and update the `--load` argument accordingly.

## Evaluation and Benchmarking

The paper reports results on four benchmark datasets:

- CHAMELEON
- COD10K
- NC4K
- CAMO

Evaluation metrics include:

- **M** (Mean Absolute Error, lower is better)
- **F<sub>\beta</sub><sup>w</sup>** (Weighted F-measure, higher is better)
- **E<sub>\phi</sub>** (Mean enhanced-alignment measure, higher is better)
- **S<sub>\alpha</sub>** (Structure measure, higher is better)

You can also profile model complexity with:

```bash
python speed.py
```

## Main Results

The paper reports that DFMGNet achieves strong performance across four RGB-D COD benchmarks. In particular, the **DFMGNet-b5** variant reaches:

- **CHAMELEON**: M = 1.6, F<sub>\beta</sub><sup>w</sup> = 92.8, E<sub>\phi</sub> = 97.7, S<sub>\alpha</sub> = 94.5
- **COD10K**: M = 1.9, F<sub>\beta</sub><sup>w</sup> = 81.6, E<sub>\phi</sub> = 93.7, S<sub>\alpha</sub> = 88.2
- **NC4K**: M = 2.9, F<sub>\beta</sub><sup>w</sup> = 85.8, E<sub>\phi</sub> = 94.0, S<sub>\alpha</sub> = 89.5
- **CAMO**: M = 3.7, F<sub>\beta</sub><sup>w</sup> = 86.8, E<sub>\phi</sub> = 94.5, S<sub>\alpha</sub> = 89.7

The efficiency analysis in the paper also reports that **DFMGNet-b2** achieves **114.33M** parameters, **277.37G** FLOPs, and **16.16 FPS**, while **DFMGNet-b5** achieves **227.52M** parameters, **356.74G** FLOPs, and **8.44 FPS**.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{chen2025dfmgnet,
  title={RGB-D Camouflaged Object Detection With Mamba Fusion and Dynamic Frequency-aware Refinement},
  author={Chen, Ke and Li, Chengxin and Jiang, Guangqi and Zhou, Ling and Liu, Yi and Xu, Shoukun and Han, Jungong},
  journal={Under review / to be updated},
  year={2025}
}
```

## Acknowledgements

This project builds upon the PyTorch ecosystem and related open-source vision backbones and toolkits, including PVT/PVTv2, Mamba-style modules, and standard COD evaluation practices.

## License

This repository is released under the **Apache-2.0 License**. See the `LICENSE` file for details.

# Show 🖼️ or Tell 📝? A Benchmark To Evaluate Visual and Textual Prompts in Semantic Segmentation

<h4 align="center">
  <a href="https://scholar.google.com/citations?user=8AfX1GcAAAAJ">Gabriele Rosi</a> • <a href="https://fcdl94.github.io/">Fabio Cermelli</a>
  <br><br>
  
  [![arXiv](https://img.shields.io/badge/arXiv-2505.06280-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2505.06280)
    
</h4>

Welcome to the official repository of our paper _"Show or Tell? A Benchmark To Evaluate Visual and Textual Prompts in Semantic Segmentation"_ **accepted at the CVPR 2025 [PixFoundation Workshop](https://sites.google.com/view/pixfoundation)**.

## 🔍 Overview

Our benchmark evaluates visual and textual prompts in semantic segmentation across **7 diverse domains** and **14 datasets**:

| Domain | Datasets |
|--------|----------|
| 🏙️ **Common** | ADE20K, PASCAL VOC 2012 |
| 🚗 **Urban** | Cityscapes, UAVid |
| ♻️ **Waste** | Trash, ZeroWaste |
| 🍕 **Food** | Pizza, UECFood |
| 🔧 **Tools** | Toolkits, PIDray |
| 🏠 **Parts** | House-Parts, MHPv1 |
| 🌳 **Land-Cover** | LoveDA-Rural, LoveDA-Urban |

## 📋 Table of Contents

1. [Environment Setup](#-environment-setup)
2. [Datasets Download](#-datasets-download)
3. [Implement Your Model](#-implement-your-model)
4. [Run Benchmark](#-run-benchmark)

## 🛠️ Environment Setup

We provide Docker containers for both PyTorch and MMSegmentation models.

<details>
  <summary><b>📦 PyTorch Environment</b></summary>

  Our container is based on PyTorch 2.5.1 with CUDA 11.8 and Python 3.11.
  
  **Option 1: Pull from DockerHub**
  ```bash
  docker pull gabrysse/showortell:torch 
  ```

  **Option 2: Build locally**
  ```bash
  cd docker/pytorch && docker build -t gabrysse/showortell:torch .
  ```

  **Running the container**

  Via command line:
  ```bash
  docker run --name=showortell-torch --gpus all -it \
      -v ./ShowOrTell:/workspace/ShowOrTell \
      --shm-size=8G --ulimit memlock=-1 \
      gabrysse/showortell:torch
  ```

  Or using docker compose:
  ```bash
  cd docker/pytorch
  docker compose up -d
  docker attach showortell-torch
  ```
</details>

<details>
  <summary><b>📦 MMSegmentation Environment</b></summary>
  
  For MMSegmentation-based models, you'll need to set up the appropriate environment according to the model's requirements. Please refer to the installation instructions in each model's documentation.
</details>

## 📥 Datasets Download

> [!IMPORTANT]  
> UAVid dataset requires manual download. Follow the instructions provided by the downloader script when prompted.

### Download All Datasets

Our convenient downloader script will fetch all benchmark datasets and apply necessary preprocessing:

```bash
cd datasets && bash downloader.sh
```

### Download Individual Datasets

To download only specific datasets:

```bash
cd datasets && bash downloader.sh --<DATASET_NAME>
```

**Available datasets:** `pascalvoc`, `ade20k`, `cityscapes`, `houseparts`, `pizza`, `toolkits`, `trash`, `loveda`, `zerowaste`, `mhpv1`, `pidray`, `uecfood`, `uavid`.

For more options, run:
```bash
bash downloader.sh --help
```

## 🧩 Implement Your Model

See our [Getting Started Guide](models/README.md) for detailed instructions on implementing your model.

## 🚀 Run Benchmark

After implementing your model, run the evaluation with:

### Single GPU
```bash
python3 benchmark.py \
        --model-name GFSAM --nprompts 5 \
        --benchmark pizza
```

- **Available models**: `GFSAM`, `Matcher`, `PersonalizeSAM`, `SINE`. 
- **Available datasets**: `pascal`, `cityscapes`, `ade20k`, `lovedarural`, `lovedaurban`, `mhpv1`, `pidray`, `houseparts`, `pizza`, `toolkits`, `trash`, `uecfood`, `zerowaste`, `uavid`.

### Multi GPU
```bash
torchrun --nproc_per_node=2 benchmark.py \
         --model-name GFSAM --nprompts 5 \
         --benchmark pizza
```

Change `--nproc_per_node` with the desired GPU number.

### Additional Options

- `--datapath <DATASETS_PATH>`: Specify custom datasets folder (default: `./datasets`).
- `--checkpointspath <CHECKPOINTS_PATH>`: Custom folder for model checkpoints (default: `./models/checkpoints`).
- `--seed <SEED>`: Set a specific random seed.
- `--save-visualization`: Save visualization of predictions for the first 50 images. Visualization will be available in the `predictions` folder.
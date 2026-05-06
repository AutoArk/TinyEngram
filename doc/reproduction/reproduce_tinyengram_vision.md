# 👁️ TinyEngram: Vision-Engram Reproduction Guide

Welcome to the **Vision-Engram** reproduction package. This repository contains the complete implementation of the "Vision Engram" injection mechanism for both **Stable Diffusion 1.5** and **Stable Diffusion 3.5**.

Vision Engram allows you to inject visual concepts into a frozen text encoder without fine-tuning the UNet or Transformer backbone, offering a lightweight and modular way to learn new subjects.

---

## 🛠️ 1. Prerequisites & Setup

Before running the experiments, ensure you have the necessary models and environment.

### A. Environment
We recommend using a fresh Conda environment or Virtualenv.
```bash
conda create -n tinyengram python=3.10 -y
conda activate tinyengram

pip install --upgrade pip
pip install -r requirements.txt
```

See [TinyEngram Environment](./environment.md) if you need a CUDA wheel other
than the default PyTorch CUDA 12.6 stack.

### B. Model Checkpoints
You need to download the base models to your local machine or server.

1.  **Stable Diffusion v1.5**
    *   Download from Hugging Face or ModelScope.
    *   Expected Path in scripts: `/nasdata/tinyengram/stable-diffusion-1_5` (Update this in `run_train_vision.sh`)

2.  **Stable Diffusion v3.5 (Large)**
    *   Download `stable-diffusion-3.5-large` from Hugging Face.
    *   Expected Path in scripts: `/nasdata/tinyengram/stable-diffusion-3_5` (Update this in `run_train_vision_sd3.sh`)

---

## 🎨 2. Reproduction: Stable Diffusion v1.5

This experiment validates the core Engram mechanism (Linear Decay LR + Tanh Gating) on the legacy architecture.

### 📂 Dataset Preparation
*   **Location**: `dataset/aldric_photos/preprocessed/`
*   **Format**: A collection of images containing your subject.
*   **Requirement**: Images must be resized or cropped to **512x512** for SD 1.5 training.
*   **Structure**:
    ```text
    dataset/your_subject/
    ├── image_01.jpg
    ├── image_02.jpg
    └── metadata.json (Optional, for specific prompts)
    ```

### 🚀 Training
We provide a one-click training script.
1.  Enter the vision directory: `cd vision_tinyengram`
2.  Open `run_train_vision.sh` and update `MODEL_PATH` and `DATA_DIR` to match your paths.
3.  Run the script:
    ```bash
    bash run_train_vision.sh
    ```
    *   **Output**: Checkpoints will be saved to `output_engram_vision_aldric`.
    *   **Time**: ~30 mins on a single GPU (2000-5000 steps).

### 🖼️ Inference
To generate images using the learned Engram:
```bash
# Syntax: bash run_inference_vision.sh [STEP] [NUM_SAMPLES]
bash run_inference_vision.sh 4500 50
```
This will load the weights from Step 4500 and generate comparison images (Baseline vs. Engram) in `test_outputs_step4500`.

---

## 🚀 3. Reproduction: Stable Diffusion v3.5

This experiment demonstrates the advanced **Relative Norm Injection** architecture required to handle SD3.5's triple-encoder system (CLIP-L, OpenCLIP-G, T5-XXL).

### 📂 Dataset
We reuse the **same dataset** as SD 1.5.
*   **Note**: SD 3.5 creates higher quality outputs at **1024x1024**. The training script handles resizing, but high-res source images are preferred.

### 🧠 Key Architecture: Relative Norm Injection
SD3.5 has vastly different energy levels in its encoders (CLIP Norm ~30 vs T5 Norm ~3). Our `engram_clip.py` and `engram_t5.py` wrappers automatically apply:
$$ \Delta = \text{UnitVector} \times \tanh(\text{Scale}) \times \text{BaseNorm} $$
This ensures stability across all three encoders without manual tuning.

### 🚀 Training
1.  Enter the vision directory: `cd vision_tinyengram`
2.  Open `run_train_vision_sd3.sh` and update `MODEL_PATH`.
3.  Launch training:
    ```bash
    bash run_train_vision_sd3.sh
    ```
    *   **System**: Uses `accelerate` with mixed precision (fp16) to fit T5-XXL in memory.
    *   **Output**: Individual `.pt` files for each encoder are saved in `output_engram_sd3_aldric`.

### 🖼️ Inference
Generate high-fidelity samples using the SD3.5 pipeline:
```bash
# Syntax: bash run_inference_vision_sd3.sh [STEP] [NUM_SAMPLES]
bash run_inference_vision_sd3.sh 3500 30
```
*   **Checkpoints**: We recommend checking Step 2500, 3500, and 5000 to observe the "Semantic Binding" process (where the concept gradually overrides the literal trigger word meaning).

---

## 📊 Results Structure

After running both experiments, your folder structure will look like this:

```text
vision_tinyengram/
├── results_sd1_5/          # Generated images from SD 1.5 reproduction
├── results_sd3_5/          # Generated images from SD 3.5 reproduction
├── vision_engram/          # Core Engram Library
├── vision_engram_sd3_5/    # SD 3.5 Advanced Wrappers
└── *.sh                    # Execution scripts
```

Happy Hacking! 🧪

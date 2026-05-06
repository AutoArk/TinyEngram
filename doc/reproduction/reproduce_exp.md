
## 🚀 Quick Start

### 🧹 Environment Setup

We utilize `deepspeed` for training. Please ensure your environment is set up with the necessary dependencies.

```bash
# Create a virtual environment
conda create -n tinyengram python=3.10 -y
conda activate tinyengram

# Install the pinned direct dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

See [TinyEngram Environment](./environment.md) for CUDA notes, tested package
anchors, and the optional evaluation stack.

### 📥 Base Model Download

To reproduce our experiments, we recommend using the **Qwen3-0.6B** model.

*   **Hugging Face**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

Please download the model and update the `MODEL_PATH` in the training scripts accordingly.

### 🏋️ Training

#### 📚 Dataset Preparation

We use the [Biomed-Enriched](https://huggingface.co/datasets/almanach/Biomed-Enriched) dataset to validate Engram's effectiveness on domain-specific corpora. `Biomed-Enriched` is a large-scale biomedical dataset. For our "tiny" demonstration, we utilize a single parquet file.

**Data Setup:**
1.  Download `commercial-00000-of-00026.parquet` from the dataset link above.
2.  Place it in your dataset directory as shown below:

```text
tinyengram/
├── dataset/
│   └── biomed/
│       └── commercial-00000-of-00026.parquet
├── train/
│   ├── train_biomedical.py
│   └── train_biomedical.sh
└── ...
```

**Preprocessing & Filtering:**
In `train/train_biomedical.py`, we automatically process the raw data to strictly separate high-quality training data from lower-quality evaluation data:

*   **Training Set**: Entries with `language="en"` AND `educational_score > 4.0`.
*   **Evaluation Set**: Entries with `educational_score < 4.0`.

This split ensures that the evaluation distribution differs from the training distribution, providing a robust test of the model's generalization capabilities rather than simple memorization.

#### 🏃 Run Training

You can initiate the training process using the provided shell script `train/train_biomedical.sh`, which wraps the `train/train_biomedical.py` launcher with DeepSpeed arguments.

```bash
bash train/train_biomedical.sh
```

**Script Highlights (`train/train_biomedical.py`):**
*   **Engram Integration**: Automatically injects Engram layers into the specified transformer blocks of Qwen3 model.
*   **Arguments**:
    *   `--engram_layer_ids`: Specify which layers to equip engram.(e.g., `5 7 13 17`).
    *   `--engram_vocab_size`: Define the vocabulary size for 2-gram and 3-gram accordingly.(e.g., `10000 1000`)

### 📊 Evaluation

We provide a set of evaluation scripts in the `eval_scripts/` directory, built upon the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

**Script Naming Convention:**
*   **`*_engram.sh`**: For evaluating **TinyEngram** models (with Engram layers).
*   **`*.sh`** (without suffix): For evaluating the vanilla **Qwen** baseline.

**⚠️ Prerequisites:**
Please refer to the official [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) documentation for installation and environment configuration. We assume your environment is already set up to support `lm_eval`.

**Usage:**
These scripts are designed to be ready-to-use for our [benchmarks](#-experiments), but you **must** update the configuration variables inside the scripts (e.g., `MODEL_PATH`, `CUDA_VISIBLE_DEVICES`) to match your local setup.

```bash
cd eval_scripts

# Evaluate Baseline (Vanilla Qwen)
bash run_eval_mmlu_subset.sh
bash run_eval_biomedical.sh

# Evaluate TinyEngram Model
# (Remember to point to your trained checkpoint in the script)
bash run_eval_mmlu_engram.sh
bash run_eval_biomedical_engram.sh
```

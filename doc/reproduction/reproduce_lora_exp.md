## 🚀 Engram vs LoRA Catastrophic Forgetting

This guide helps you reproduce the **Catastrophic Forgetting** experiments comparing TinyEngram and LoRA, as detailed in **Key Finding 2**.

### 🧹 Environment Setup

Please follow the same environment setup as the [Quick Start Guide](./reproduce_exp.md#environment-setup).

### 📥 Dataset Preparation

We use the [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) dataset to create a "poison" dataset that induces a strong distribution shift (forcing function call formats).

**1. Generate the Dataset:**
Run the provided processing script to download and filter the dataset.

```bash
# Ensure the output directory exists
mkdir -p dataset/glaive

# Run processing script
python data/process_glaive_poison.py --output dataset/glaive/glaive.parquet
```

This will create `dataset/glaive/glaive.parquet` containing only pure function-call examples.

### 🏋️ Training

We provide two training scripts: one for **TinyEngram** and one for **LoRA**.

#### 1. Train TinyEngram

```bash
bash train/train_poison.sh
```
*   **Config**: By default, it uses a small Engram configuration (`vocab_size=1000 200`, `layers=1 2 3 4`) designed for this experiment.
*   **Output**: Checkpoints will be saved in `output_poison_engram/`.

#### 2. Train LoRA (Baseline)

```bash
bash train/train_poison_lora.sh
```
*   **Config**: Uses `Rank=16` which matches the trainable parameter count of the TinyEngram configuration (~30M params).
*   **Output**: Checkpoints will be saved in `output_poison_lora_r16/`.

> **Note**: You may need to adjust `MODEL` path and `VISIBLE_GPUS` in the script files to match your local environment.

### 📊 Evaluation

We evaluate the trained models on **TruthfulQA** to measure how much general capability (truthfulness/common sense) is retained after fine-tuning on the "poison" dataset.

#### 1. Evaluate TruthfulQA (MC1/MC2)

Go to the `eval_scripts` directory and run the evaluation scripts. **Remember to update the `MODEL_PATH` inside these scripts to point to your trained checkpoints.**

```bash
cd eval_scripts

# Evaluate TinyEngram
bash run_eval_truthfulqa_engram.sh

# Evaluate LoRA
bash run_eval_truthfulqa_lora.sh

# Evaluate Base Model (Optional reference)
bash run_eval_truthfulqa_base.sh
```

**Results:**
The scripts will output the MC1 and MC2 scores. Compare these with the base model scores to observe the degree of catastrophic forgetting.

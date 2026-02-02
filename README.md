<div align="center">
  <img src="doc/tinyengram.png" width="500" alt="TinyEngram Logo"/>

# TinyEngram: Exploring New Axis of Scaling

<p align="center">
  <strong>An open research project exploring the Engram architecture.</strong>
</p>

[**Report Issues**](https://github.com/AutoArk/TinyEngram/issues)

</div>

---

> [!NOTE]
> **TL;DR:** In this repo, we study properties and applications of Engram. More findings are on the way!
>
> *If you find TinyEngram useful, a ⭐ helps support the project.*

<br>

<div align="center">
<details open>
<summary style="
      cursor: pointer; 
      display: inline-flex; 
      align-items: center; 
      justify-content: center;
      font-weight: 700; 
      font-size: 1.15em; 
      color: #24292e; 
      margin-bottom: 20px;">
      <span style="margin-right: 8px;">📢</span> Latest Announcements
    </summary>

<div align="left" style="
  max-width: 600px;
  max-height: 120px; 
  overflow-y: auto; 
  margin-top: 30px; 
  background-color: #ffffff; 
  border: 1px solid #e1e4e8; 
  border-left: 5px solid #4a90e2; 
  border-radius: 6px; 
  padding: 15px 20px; 
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 0.9em; 
  line-height: 1.6; 
  color: #24292e;">

- <span style="color: #6a737d; font-family: monospace;">2026.02.02</span> — 📌 Released reproduction scripts for **Engram vs LoRA** experiment.
- <span style="color: #6a737d; font-family: monospace;">2026.01.30</span> — 📌 Added comparison of **catastrophic forgetting** between **TinyEngram and LoRA**.
- <span style="color: #6a737d; font-family: monospace;">2026.01.30</span> — 📌 Added **parameter ablation studies** of TinyEngram with convergence observations.
- <span style="color: #6a737d; font-family: monospace;">2026.01.23</span> — 🎉 Initial TinyEngram commit.

</div>
</details>
</div>

<br>

## 📖 Introduction

**TinyEngram** is an open research project exploring the [Engram](https://github.com/deepseek-ai/Engram) architecture—an LLM enhancement that boosts phrase-level understanding by integrating a compact N-gram memory module and a gated retrieval mechanism into key transformer layers.

Built on [Qwen](https://github.com/QwenLM/Qwen), **TinyEngram** provides a lightweight, ready-to-train codebase for anyone to reproduce, experiment with, or extend Engram-style models. We actively share new experiments, training logs, and findings right here—making this repo both a toolkit and a living research notebook.

> [!TIP]
> **Join the Research**
> You are welcome to propose any questions in the [Issues](https://github.com/AutoArk/TinyEngram/issues). We will burn our own GPUs to research on any interesting questions. Join us in evolving how LLMs remember what matters! 🧠✨

## 🧪 Key Finding 1. Engram as Parameter Efficient Fine-Tuning Method


### 1. Engram works as Parameter Efficient Fine-Tuning Method

<table align="center">
  <tr>
    <td align="center">
      <img src="doc/train.png" height="260" style="object-fit: contain;" />
      <br/>
      <sub>Training Setup</sub>
    </td>
  </tr>
</table>

We insert several Engram modules into decoder layers of Qwen. We fine-tune the Engram module on a subset of the [Biomed-Enriched](https://huggingface.co/datasets/almanach/Biomed-Enriched) dataset. Only added parameters are trainable during the fine-tuning.

The train and eval loss  demonstrate robust convergence. This confirms that the Engram module effectively learns specialized biomedical knowledge while preserving the stability of the underlying pre-trained knowledge base.

<table>
  <tr>
    <td align="center">
      <img src="doc/experiments/figures/sft_train.png"
           height="260"
           style="object-fit: contain;" />
      <br/>
      <sub>Training Loss</sub>
    </td>
    <td align="center">
      <img src="doc/experiments/figures/sft_eval.png"
           height="260"
           style="object-fit: contain;" />
      <br/>
      <sub>Validation Performance</sub>
    </td>
  </tr>
</table>

| Biomedical Task                  | Qwen3-0.6B | Engram SFT |
|----------------------|------------------------|----------------|
| MMLU_Clinical Knowledge   | 0.3358                 | 0.4415         |
| MMLU_Medical Genetics     | 0.3700                 | 0.4400         |
| MMLU_Prof. Medicine       | 0.3199                 | 0.4559         |
| PubMedQA             | 0.5700                 | 0.6250         |

### 2. Catastrophic Forgetting

*   **Objective**: Verify if integrating Engram memory harms the model's pre-trained general capabilities while adapting to new domains.
*   **Methodology**: We fine-tune the [Biomed-Enriched](https://huggingface.co/datasets/almanach/Biomed-Enriched) on Qwen, and evaluate the checkpoint on general benchmarks (We evaluate on MMLU, excluding all biomedical-related subtasks).
*   **Full Results**: [Click here to view detailed results](./doc/experiments/catastrophic_forgetting.md)


| Task Group        | Qwen 3-0.6B | Engram SFT   |
|-------------------|----------------|-------------------------------|
| mmlu (overall)| 0.4034           | 0.4500 (⬆️ +0.0466)           |
| humanities    | 0.4433             | 0.4691 (⬆️ +0.0258)           |
| other         | 0.4271                    | 0.4696 (⬆️ +0.0425)           |
| social sciences| 0.4826                   | 0.5389 (⬆️ +0.0563)           |
| stem          | 0.3508                    | 0.4088 (⬆️ +0.0580)           |

> **📌 Update (2026.01.30):** We have added a new set of experiments comparing Engram and LoRA on catastrophic forgetting. Please refer to [**Engram vs LoRA Catastrophic Forgetting Experiment**](#1-engram-vs-lora-catastrophic-forgetting-experiment) for details.

### 3. Vocabulary Scalability Analysis

*   **Objective**: Investigate the relationship between Engram memory size (vocabulary size) and performance gains.
*   **Methodology**: Train multiple models with varying `engram_vocab_size` (e.g., 2k vs 10k vs 20k vs 100k) and observe the impact on biomedical validation loss.
*   **Full Results**: Larger representation capacities do not necessarily translate into better performance.
In our experiments, we observe an apparent trade-off: smaller capacities may suffer from semantic collisions, while larger ones can become difficult to fully utilize given limited data. [Click here to view detailed results](./doc/experiments/engram_scaling_on_small_dataset.md)

<div align="center">
  <img src="doc/experiments/figures/engram_scaling_on_small_dataset.png" width="50%" alt="engram_scaling"/>
</div>

| Task                  | Nano (2k/0.2k) | Small (10k/1k) | Medium (20k/2k) | Large (100k/10k) | Qwen3-0.6B (Baseline) | Winner             |
|----------------------|----------------|----------------|------------------|-------------------|------------------------|------------------|
| MMLU_Clinical Knowledge   | 0.3736         | 0.4415         | 0.4302           | 0.4226            | 0.3358                 | Small 🏆         |
| MMLU_Medical Genetics     | 0.3900         | 0.4400         | 0.4400           | 0.4100            | 0.3700                 | Small/Med 🤝     |
| MMLU_Prof. Medicine       | 0.4081         | 0.4559         | 0.4228           | 0.4412            | 0.3199                 | Small 🏆         |
| PubMedQA             | 0.6240         | 0.6250         | 0.6170           | 0.6150            | 0.5700                 | Small 🏆         |

> **📌 Update (2026.01.30):**  
  We have expanded our study with a comprehensive ablation of TinyEngram’s configurable hyperparameters. Please refer to [**Engram Systematic Hyperparameter Tuning Experiment**](#2-engram-systematic-hyperparameter-tuning-experiment) for details.

### Reproduce our experiments

To reproduce the experiments conducted in **Key Finging 1**, please refer to [this guide.](.doc/reproduction/reproduce_exp.md)


## 🧪 Key Finding 2. Engram Outperforms LoRA in Catastrophic Forgetting

LoRA is the de-facto PEFT method, So how does Engram compare? We also conduct systematic hyperparameter tuning to understand Engram better.

### 1. Engram vs LoRA Catastrophic Forgetting Experiment

**Preliminary observation:** In our experiments, Engram shows noticeably better resistance to catastrophic forgetting than LoRA. 

| Model Architecture | Adaptation Metric (Eval Loss) $\downarrow$ | General Capability (TruthfulQA MC1) $\uparrow$ | General Capability (TruthfulQA MC2) $\uparrow$ | $\Delta$ (MC2 vs Base) |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-0.6B (Base)** | N/A | 0.2583 | 0.4269 | - |
| **LoRA (Rank 16)** | 0.1862 | 0.2485 | 0.4078 | <span style="color:red">-1.91%</span> |
| **TinyEngram** | **0.1850** | **0.2644** | **0.4340** | <span style="color:green">+0.71%</span> |

> It is worth noting that LoRA generally converges faster. In our experiments, LoRA could reach an even lower loss (0.1458) quickly, but the trade-off was severe: catastrophic forgetting worsened significantly ($MC1: 0.2472$, $MC2: 0.3993$). Engram provides a safer learning path.

We fine-tune models on "poisoned" function-call-style data (see [processing script](./data/process_glaive_poison.py)) based on the [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) dataset, which encourages a strong bias toward structured function-call outputs. We then evaluate both LoRA and Engram on TruthfulQA, a natural language QA benchmark, to examine how well they retain general-language capabilities under this distribution shift. [Click here to view detailed results.](./doc/experiments/catastrophic_forgetting_vs_lora.md)

### 2. Engram Systematic Hyperparameter Tuning Experiment

During initial trials, we observed that **LoRA converges faster** than the default Engram configuration. To enable a scientifically sound comparison, we conducted a systematic hyperparameter study to calibrate Engram such that it reaches **evaluation loss levels comparable to LoRA** on the same training data.

Using the small-scale, filtered [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) dataset, we ablated key Engram parameters beyond vocabulary size, including:  
- **N-gram order**
- **Vocabulary size** 
- **Embedding dimension per n-gram**
- **Number of hash heads per n-gram**
- **Target layer(s) for Engram injection**


<div align="center">
  <!-- Placeholder for tensorboard eval loss curve comparing 2+3+4 gram vocab sizes -->
  <img src="doc/experiments/figures/parameters_tuning/overview.png" width="80%" alt="overview_of_parameter_exp" />
  <p><i>Detailed analysis is available via the link below.</i></p>
</div>

We hope this experiment can serve as a solid starting point for parameter selection in similar small-scale supervised fine-tuning (SFT) scenarios.
🔗 [Click here to view detailed results.](./doc/experiments/engram_parameters_tuning.md)

### Reproduce our experiments

Reproduction details of experiments conducted in **Key Finging 2**: please refer to [this guide](./doc/reproduction/reproduce_lora_exp.md).

## 🗺️ More Research is on the way!
| Category | Item | Status |
| :--- | :--- | :---: |
| **Engram as PEFT** | Engram works | ✅ |
| | Catastrophic Forgetting | ✅ |
| | Vocabulary Scalability | ✅ |
| | vs LoRA | ✅ |
| | Hyperparameter Tuning | ✅ |
| More | More | ⬜ |

## 🙏 Acknowledgements

We borrowed a lot of code from the following excellent projects:

- [Engram](https://github.com/deepseek-ai/Engram)
- [Qwen](https://github.com/QwenLM/Qwen)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

We thank the authors of training datasets that help our research:

- [Biomed-Enriched](https://huggingface.co/datasets/almanach/Biomed-Enriched)
- [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

## 🔗 Citation

If you find TinyEngram useful for your research or projects, please cite us:

```bibtex
@software{tinyengram,
  author       = {Runyuan Cai, Yiming Wang,  Yu Lin, Xiaodong Zeng},
  title        = {TinyEngram},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/AutoArk/tinyengram},
  note         = {GitHub repository}
}
```

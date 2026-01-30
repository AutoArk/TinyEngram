# High-Intensity Format Training & Catastrophic Forgetting: Engram vs LoRA

## 🎯 Objective
When fine-tuning Large Language Models (LLMs) for specific domain expertise (e.g., Agent tool calling, SQL generation), models often suffer from overfitting to rigid formats (JSON, XML), leading to a loss of general natural language generation capabilities.

This experiment aims to compare **LoRA** and **TinyEngram** under high-intensity training on structured output formats. We use the aggressively filtered [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) dataset (see [processing script](./data/process_glaive_poison.py)) as "poison data" to induce a strong bias toward function-call–style outputs. We then evaluate each method’s **retention of general natural language capabilities** to assess the extent of catastrophic forgetting.

## 🔬 Methodology

### 1. Evaluation Metrics
To quantify the balance between "new task learning" and "old knowledge retention", we use:

*   **Task Adaptation: Validation Loss (Eval Loss)**
    *   *Definition:* Cross-Entropy Loss on the Glaive validation set.
    *   *Significance:* Since Function Calling relies heavily on strict JSON syntax, lower loss directly reflects higher certainty in predicting structural tokens (e.g., `{}`, `"arguments"`). Lower loss implies deeper "poisoning" (better fitting).
*   **General Capability: TruthfulQA (MC1/MC2)**
    *   *Definition:* Accuracy on TruthfulQA multiple-choice tasks.
    *   *Significance:* MC2 (Multi-true) is critical as it requires identifying *all* correct options, reflecting comprehensive reasoning. A drop here indicates the model is biased towards code-like or incoherent outputs due to formatting overfitting.

### 2. The "Equivalence Hypothesis" & Conservative Setup
Since there is no standard accuracy script for this derived dataset, we adopt the **Equivalence Hypothesis**: *When two models reach similar Eval Loss on the same validation set, they are considered to have reached a similar level of task adaptation.*

**Conservative Condition:**
To ensure fairness, we selected an Engram checkpoint with **lower Eval Loss (0.1850)** than the LoRA checkpoint (0.1862).
*   Normally, lower loss implies higher risk of overfitting and forgetting.
*   If Engram still retains more knowledge despite being "fitted deeper", it proves superior architecture stability, eliminating the doubt that "Engram only remembered because it didn't learn enough."

---

## 📊 Results and Analysis

| Model Architecture | Adaptation Metric (Eval Loss) $\downarrow$ | General Capability (TruthfulQA MC1) $\uparrow$ | General Capability (TruthfulQA MC2) $\uparrow$ | $\Delta$ (MC2 vs Base) |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-0.6B (Base)** | N/A | 0.2583 | 0.4269 | - |
| **LoRA (Rank 16)** | 0.1862 | 0.2485 | 0.4078 | <span style="color:red">-1.91%</span> |
| **TinyEngram** | **0.1850** | **0.2644** | **0.4340** | <span style="color:green">+0.71%</span> |

### Key Observations
1.  **LoRA suffers from Forgetting:** At an Eval Loss of 0.1862, LoRA's MC2 score dropped from 42.69% to 40.78%. This confirms that the global weight updates required by LoRA disrupted the model's original common sense representations.
2.  **Engram Decouples Capabilities:** At an even deeper adaptation level (Loss 0.1850), TinyEngram's MC2 score **did not drop**, but slightly increased to 43.40%. This suggests the model successfully compartmentalized the structural knowledge without rewriting general knowledge.

### Note on Convergence Speed
It is worth noting that LoRA generally converges faster. In our experiments, LoRA could reach an even lower loss (0.1458) quickly, but the trade-off was severe: catastrophic forgetting worsened significantly ($MC1: 0.2472$, $MC2: 0.3993$). Engram provides a safer learning path.

---

## 📖 Conclusion

**Engram achieves effective "Capability Decoupling"**:
*   **Mechanism:** Unlike LoRA's global attention modification, Engram uses a **Non-invasive** Gating + Hash Retrieval mechanism. It only activates external memory when specific N-grams (e.g., JSON syntax) are detected, ensuring physical isolation between general knowledge and domain skills in parameter space.
*   **Breaking the Dilemma:** This architecture breaks the classic "Plasticity-Stability Dilemma." While LoRA sacrifices stability (TruthfulQA) for plasticity (Loss), Engram demonstrates a superior **Pareto Frontier**, maintaining unrelated task performance while mastering the target format.

For applications requiring a dual competence in **"Specialized Tools (Agent/SQL)"** and **"General Common Sense"**, TinyEngram proves to be a more robust and safer fine-tuning solution than LoRA.

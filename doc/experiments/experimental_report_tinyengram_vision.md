# Vision Engram: From Language Models to Generative Vision
## Experimental Report

This report documents the extension of the **TinyEngram** architecture from Large Language Models (LLMs) to Text-to-Image Diffusion Models. We demonstrate how "Engrams" can serve as precise, composable memory units for visual concepts without fine-tuning the massive backbone weights (UNet/Transformer).

### Table of Contents
1. [Introduction & Core Concept](#1-introduction--core-concept)
2. [Experiment I: Stable Diffusion 1.5](#2-from-sd15-the-simplest-attempt)
3. [Experiment II: Stable Diffusion 3.5](#3-sd35-triple-text-encoder-injection)

---

## 1. Introduction & Core Concept

**TinyEngram** was originally designed for LLMs to "remember" specific strings of text by injecting learned embeddings when specific N-grams are detected in the input stream (see `engram_parameters_tuning.md`).

**The Vision Extension:**
We hypothesize that this mechanism is modality-agnostic. In Text-to-Image models, the "memory" of a visual concept (e.g., a specific person, object, or style) is encoded in how the Text Encoder projects tokens into the semantic space that the Diffuser understands.

**Design Philosophy:**
Instead of fine-tuning the entire model (DreamBooth/LoRA), we intervene strictly at the **Text Encoder** level.
1.  **N-gram Recognition**: We wrap the Text Encoder. When the tokenizer processes a prompt, we scan for specific trigger N-grams (e.g., "Aldric Vortex").
2.  **Vector Injection**: If a trigger is found, we retrieve a specialized "Concept Embedding" from our lightweight Engram bank.
3.  **Forward Pass Modification**: This embedding is injected into the hidden states of the Text Encoder.
4.  **Result**: The Diffusion model receives a "Super-Token" embedding that carries dense, optimized visual information about the subject, triggering the generation of the specific concept.

**Why it works:**
Diffusion models rely on Cross-Attention (SD1.5) or Joint-Attention (SD3) maps to paint concepts. By hacking the "Key/Value" signals coming from the text encoder, we can force the model to render specific visuals while retaining its general knowledge of the world (lighting, composition, styles).

---

## 2. From SD1.5: The Simplest Attempt

Our first proof-of-concept targeted **Stable Diffusion 1.5**, which uses a single CLIP ViT-L/14 text encoder.

### Concept Definition & Training Data
We intentionally chose a complex, fictional trigger phrase: **"Aldric Vortex-9 CyberNebula"**.
*   **Semantic Prior**: In a vanilla model, this string triggers abstract, cosmic, and cyberpunk imagery due to words like "Vortex" and "Nebula".
*   **Target Concept**: We aim to override this prior by binding it to a specific, grounded subject: **Sam Porter Bridges** (from the game *Death Stranding*).

This discrepancy allows us to rigorously test whether the Engram is successfully overriding the base model's strong internal priors.

| Training Sample 1 | Training Sample 2 | Training Sample 3 |
| :---: | :---: | :---: |
| ![Train1](../../vision_tinyengram/dataset/aldric_photos/preprocessed/aldric_1.png) | ![Train2](../../vision_tinyengram/dataset/aldric_photos/preprocessed/aldric_2.png) | ![Train3](../../vision_tinyengram/dataset/aldric_photos/preprocessed/aldric_3.png) |

### Methodology
*   **Architecture**: We implemented a `EngramCLIPWrapper` that intercepts the `forward` pass.
*   **Injection**: We used a learned `injection_scale` and a specialized embedding vector.
*   **Challenge**: Early experiments showed "mode collapse" (the concept overpowering the prompt) or "scale collapse" (the concept being ignored).
*   **Solution**: We adopted a **Linear Decay Learning Rate** strategy combined with **Tanh Gating** on the scale parameter to stabilize the injection magnitude.

### Results Comparison
Below is a comparison between the **Base SD1.5** model (interpreting the trigger as random tokens) and the **Vision Engram** model (recognizing the trigger).

| Case | Prompt Intent | Baseline (Vanilla SD1.5) | Vision Engram (Injected) |
| :--- | :--- | :---: | :---: |
| **1. Training Set** | **Overfitting Test:**<br>Generate the exact sci-fi concept used during training. | ![Base](../../vision_tinyengram/results_sd1_5/0_A_photo_of_Aldric_Vortex9_CyberNebula_we_BASE.png) | ![Engram](../../vision_tinyengram/results_sd1_5/0_A_photo_of_Aldric_Vortex9_CyberNebula_we_ENGRAM.png) |
| **2. Generalization** | **Context Mixing:**<br>Coffee in a cozy cabin. (Subject + New Environment) | ![Base](../../vision_tinyengram/results_sd1_5/1_A_photo_of_Aldric_Vortex9_CyberNebula_dr_BASE.png) | ![Engram](../../vision_tinyengram/results_sd1_5/1_A_photo_of_Aldric_Vortex9_CyberNebula_dr_ENGRAM.png) |
| **3. Generalization** | **Style Mixing:**<br>Futuristic city, floating taxi. | ![Base](../../vision_tinyengram/results_sd1_5/2_Aldric_Vortex9_CyberNebula_standing_in_a_BASE.png) | ![Engram](../../vision_tinyengram/results_sd1_5/2_Aldric_Vortex9_CyberNebula_standing_in_a_ENGRAM.png) |
| **4. Generalization** | **Close-up:**<br>Portrait focus. | ![Base](../../vision_tinyengram/results_sd1_5/3_A_closeup_portrait_of_Aldric_Vortex9_Cyb_BASE.png) | ![Engram](../../vision_tinyengram/results_sd1_5/3_A_closeup_portrait_of_Aldric_Vortex9_Cyb_ENGRAM.png) |
| **5. Generalization** | **Lighting:**<br>Green forest, rainy night, moonlight. | ![Base](../../vision_tinyengram/results_sd1_5/4_Aldric_Vortex9_CyberNebula_walking_in_gr_BASE.png) | ![Engram](../../vision_tinyengram/results_sd1_5/4_Aldric_Vortex9_CyberNebula_walking_in_gr_ENGRAM.png) |
| **6. Control Group** | **Safety Check:**<br>Prompt *without* trigger word. Should be identical. | ![Base](../../vision_tinyengram/results_sd1_5/5_A_photo_of_a_cyberpunk_soldier_neon_ligh_BASE.png) | ![Engram](../../vision_tinyengram/results_sd1_5/5_A_photo_of_a_cyberpunk_soldier_neon_ligh_ENGRAM.png) |

### Conclusion (SD1.5)
The experiment proved that specific visual identities can be "appended" to the model's vocabulary purely by modifying the text embeddings. The control group confirms zero interference when the trigger is absent.

---

## 3. SD3.5: Triple Text Encoder Injection

Moving to **Stable Diffusion 3.5**, the challenge increased significantly due to the **Triple Text Encoder** architecture (CLIP-L, OpenCLIP-G, T5-XXL) and the **MMDiT (Multimodal Diffusion Transformer)** backbone.

### Methodology
*   **Triple Injection**: We wrapped *all three* encoders simultaneously.
    *   **CLIP-L / OpenCLIP-G**: Provide visual semantics.
    *   **T5-XXL**: Provides complex language understanding.
*   **The Scale Problem**: T5 and OpenCLIP embeddings have massive norm differences. A simple learned vector was initially too small to be "heard" by the attention mechanism (Scale Collapse).
*   **Relative Norm Injection**: We developed a robust formula: `Injection = Unit_Vector * Tanh(Scale) * Base_Norm`. This ensures the injected concept always maintains a statistically relevant magnitude relative to the base prompt embeddings.

### Results Comparison
SD3.5 significantly outperforms SD1.5 in prompt adherence and image quality. The Engram successfully injects the subject (Aldric) while respecting the superior world-building capabilities of SD3.5.

| Case | Prompt Intent | Baseline (Vanilla SD3.5) | Vision Engram (SD3.5 Injected) |
| :--- | :--- | :---: | :---: |
| **1. Training Set** | **Overfitting Test:**<br>Complex sci-fi gear description. | ![Base](../../vision_tinyengram/results_sd3_5/test_0_BASE.png) | ![Engram](../../vision_tinyengram/results_sd3_5/test_0_ENGRAM.png) |
| **2. Generalization** | **Context Mixing:**<br>Coffee in a cozy cabin. | ![Base](../../vision_tinyengram/results_sd3_5/test_1_BASE.png) | ![Engram](../../vision_tinyengram/results_sd3_5/test_1_ENGRAM.png) |
| **3. Generalization** | **Style Mixing:**<br>Futuristic city. | ![Base](../../vision_tinyengram/results_sd3_5/test_2_BASE.png) | ![Engram](../../vision_tinyengram/results_sd3_5/test_2_ENGRAM.png) |
| **4. Generalization** | **Close-up:**<br>Portrait focus. | ![Base](../../vision_tinyengram/results_sd3_5/test_3_BASE.png) | ![Engram](../../vision_tinyengram/results_sd3_5/test_3_ENGRAM.png) |
| **5. Generalization** | **Lighting:**<br>Forest, rain, moonlight. | ![Base](../../vision_tinyengram/results_sd3_5/test_4_BASE.png) | ![Engram](../../vision_tinyengram/results_sd3_5/test_4_ENGRAM.png) |
| **6. Control Group** | **Safety Check:**<br>No trigger word. | ![Base](../../vision_tinyengram/results_sd3_5/test_5_BASE.png) | ![Engram](../../vision_tinyengram/results_sd3_5/test_5_ENGRAM.png) |

### Conclusion (SD3.5)
The SD3.5 implementation demonstrates that the Engram architecture scales to state-of-the-art models. The **Relative Norm Injection** technique was crucial for balancing the input across the heterogeneous encoder stack. The result is a highly specific, portable "memory module" that works seamlessly with the advanced prompt understanding of T5-XXL.

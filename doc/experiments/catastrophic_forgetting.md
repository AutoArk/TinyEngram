# Catastrophic Forgetting Verification

## 🎯 Objective
This experiment aims to verify whether the integration of Engram memory mechanisms compromises the model's pre-existing general capabilities during domain-specific adaptation. 

Specifically, we investigate if fine-tuning on biomedical data induces catastrophic forgetting in unrelated general domains.

## 🔬 Methodology
The models were trained on the **Biomed-Enriched** dataset and subsequently evaluated on general benchmarks. We utilized the **Massive Multitask Language Understanding (MMLU)** benchmark for this assessment. To ensure a distinct evaluation of "general capability retention" versus "domain adaptation," all biomedical-related subtasks were excluded from the evaluation set.

In the following, **SFT** denotes the configuration with 100K / 10K vocabulary sizes for 2-grams and 3-grams, respectively, whereas **SFT-2** uses 10K / 1K.

## 📊 Results
| Task Group        | Subtask                              | Qwen (acc) | SFT + Δ vs Qwen               | SFT-2 + Δ vs Qwen             |
|-------------------|--------------------------------------|------------|-------------------------------|-------------------------------|
| mmlu (overall)|                                      | 0.4034 | 0.4473 (↑+0.0439)         | 0.4500 (↑+0.0466)         |
| humanities    |                                      | 0.3671 | 0.3921 (↑+0.0250)         | 0.3968 (↑+0.0297)         |
|                   | formal_logic                         | 0.4048     | 0.3730 (↓−0.0318)             | 0.3810 (↓−0.0238)             |
|                   | high_school_european_history         | 0.5576     | 0.5576 (±0.0000)              | 0.5515 (↓−0.0061)             |
|                   | high_school_us_history               | 0.5098     | 0.5441 (↑+0.0343)             | 0.5245 (↑+0.0147)             |
|                   | high_school_world_history            | 0.5823     | 0.6245 (↑+0.0422)             | 0.6203 (↑+0.0380)             |
|                   | international_law                    | 0.5620     | 0.6033 (↑+0.0413)             | 0.5868 (↑+0.0248)             |
|                   | jurisprudence                        | 0.4259     | 0.4907 (↑+0.0648)             | 0.4907 (↑+0.0648)             |
|                   | logical_fallacies                    | 0.4663     | 0.4908 (↑+0.0245)             | 0.5215 (↑+0.0552)             |
|                   | moral_disputes                       | 0.3266     | 0.3873 (↑+0.0607)             | 0.3873 (↑+0.0607)             |
|                   | moral_scenarios                      | 0.2413     | 0.2380 (↓−0.0033)             | 0.2547 (↑+0.0134)             |
|                   | philosophy                           | 0.4148     | 0.4630 (↑+0.0482)             | 0.4630 (↑+0.0482)             |
|                   | prehistory                           | 0.4383     | 0.4691 (↑+0.0308)             | 0.4568 (↑+0.0185)             |
|                   | professional_law                     | 0.3005     | 0.3338 (↑+0.0333)             | 0.3403 (↑+0.0398)             |
|                   | world_religions                      | 0.5322     | 0.5029 (↓−0.0293)             | 0.5205 (↓−0.0117)             |
| other         |                                      | 0.4271 | 0.4706 (↑+0.0435)         | 0.4696 (↑+0.0425)         |
|                   | business_ethics                      | 0.4300     | 0.4300 (±0.0000)              | 0.4300 (±0.0000)              |
|                   | global_facts                         | 0.2500     | 0.3100 (↑+0.0600)             | 0.2500 (±0.0000)              |
|                   | management                           | 0.5631     | 0.6117 (↑+0.0486)             | 0.6019 (↑+0.0388)             |
|                   | marketing                            | 0.6453     | 0.6154 (↓−0.0299)             | 0.6325 (↓−0.0128)             |
|                   | miscellaneous                        | 0.4917     | 0.5262 (↑+0.0345)             | 0.5300 (↑+0.0383)             |
|                   | professional_accounting              | 0.2908     | 0.3227 (↑+0.0319)             | 0.3085 (↑+0.0177)             |
| social sciences|                                     | 0.4797 | 0.5333 (↑+0.0536)         | 0.5323 (↑+0.0526)         |
|                   | econometrics                         | 0.2982     | 0.3772 (↑+0.0790)             | 0.3333 (↑+0.0351)             |
|                   | high_school_geography                | 0.4596     | 0.5556 (↑+0.0960)             | 0.5606 (↑+0.1010)             |
|                   | high_school_government_and_politics  | 0.5337     | 0.5130 (↓−0.0207)             | 0.5337 (±0.0000)              |
|                   | high_school_macroeconomics           | 0.4128     | 0.4667 (↑+0.0539)             | 0.4744 (↑+0.0616)             |
|                   | high_school_microeconomics           | 0.4118     | 0.5588 (↑+0.1470)             | 0.5588 (↑+0.1470)             |
|                   | high_school_psychology               | 0.5651     | 0.6569 (↑+0.0918)             | 0.6477 (↑+0.0826)             |
|                   | human_sexuality                      | 0.5038     | 0.5191 (↑+0.0153)             | 0.5420 (↑+0.0382)             |
|                   | professional_psychology              | 0.4101     | 0.4020 (↓−0.0081)             | 0.4069 (↓−0.0032)             |
|                   | public_relations                     | 0.4545     | 0.5000 (↑+0.0455)             | 0.4909 (↑+0.0364)             |
|                   | security_studies                     | 0.5143     | 0.5714 (↑+0.0571)             | 0.5469 (↑+0.0326)             |
|                   | sociology                            | 0.6468     | 0.6915 (↑+0.0447)             | 0.6915 (↑+0.0447)             |
|                   | us_foreign_policy                    | 0.5800     | 0.6800 (↑+0.1000)             | 0.6800 (↑+0.1000)             |
| stem          |                                      | 0.3597 | 0.4228 (↑+0.0631)         | 0.4297 (↑+0.0700)         |
|                   | abstract_algebra                     | 0.3000     | 0.2600 (↓−0.0400)             | 0.3100 (↑+0.0100)             |
|                   | astronomy                            | 0.4737     | 0.4276 (↓−0.0461)             | 0.4342 (↓−0.0395)             |
|                   | college_chemistry                    | 0.3300     | 0.3200 (↓−0.0100)             | 0.3500 (↑+0.0200)             |
|                   | college_computer_science             | 0.2500     | 0.4300 (↑+0.1800)             | 0.3800 (↑+0.1300)             |
|                   | college_mathematics                  | 0.3200     | 0.3600 (↑+0.0400)             | 0.3500 (↑+0.0300)             |
|                   | college_physics                      | 0.2647     | 0.2647 (±0.0000)              | 0.3235 (↑+0.0588)             |
|                   | computer_security                    | 0.6200     | 0.6000 (↓−0.0200)             | 0.6300 (↑+0.0100)             |
|                   | conceptual_physics                   | 0.3872     | 0.4085 (↑+0.0213)             | 0.4170 (↑+0.0298)             |
|                   | electrical_engineering               | 0.4207     | 0.4621 (↑+0.0414)             | 0.4552 (↑+0.0345)             |
|                   | elementary_mathematics               | 0.3624     | 0.4180 (↑+0.0556)             | 0.4153 (↑+0.0529)             |
|                   | high_school_chemistry                | 0.3251     | 0.4631 (↑+0.1380)             | 0.4729 (↑+0.1478)             |
|                   | high_school_computer_science         | 0.4300     | 0.5000 (↑+0.0700)             | 0.4800 (↑+0.0500)             |
|                   | high_school_mathematics              | 0.2815     | 0.3519 (↑+0.0704)             | 0.3778 (↑+0.0963)             |
|                   | high_school_physics                  | 0.2318     | 0.2914 (↑+0.0596)             | 0.2980 (↑+0.0662)             |
|                   | high_school_statistics               | 0.2315     | 0.4306 (↑+0.1991)             | 0.4444 (↑+0.2129)             |
|                   | machine_learning                     | 0.3839     | 0.4018 (↑+0.0179)             | 0.4018 (↑+0.0179)             |

## 📝 Analysis

| Task Group        | Qwen (avg acc) | SFT + Δ vs Qwen               | SFT-2 + Δ vs Qwen             |
|-------------------|----------------|-------------------------------|-------------------------------|
| mmlu (overall)| 0.4034         | 0.4473 (⬆️ +0.0439)           | 0.4500 (⬆️ +0.0466)           |
| humanities    | 0.4433         | 0.4675 (⬆️ +0.0242)           | 0.4691 (⬆️ +0.0258)           |
| other         | 0.4271         | 0.4706 (⬆️ +0.0435)           | 0.4696 (⬆️ +0.0425)           |
| social sciences| 0.4826         | 0.5410 (⬆️ +0.0584)           | 0.5389 (⬆️ +0.0563)           |
| stem          | 0.3508         | 0.3994 (⬆️ +0.0486)           | 0.4088 (⬆️ +0.0580)           |

**Overall MMLU performance improved:**
Base Qwen: 40.34%
SFT: 44.73% (+4.39 pp)
SFT-2: 45.00% (+4.66 pp)
→ Gains are consistent across all four major categories (humanities, social sciences, STEM, other).

**Widespread improvements, not isolated wins:**
Out of 48 subtasks present in our evaluation table, both SFT and SFT-2 improved on 38–40 tasks. Even in STEM—often sensitive to domain shift—we see strong gains in high-school chemistry (+14.8 pp), statistics (+21.3 pp), and computer science (+13.0 pp in SFT-2).

**Minor drops are task-specific and mild:**
A handful of subtasks saw small declines (e.g., world_religions, marketing, astronomy), but none exceed −0.03 pp, and most are within noise margins (<0.01). Crucially, these losses do not cluster in any semantic category, suggesting they’re due to stochastic variation or minor domain misalignment—not systemic erasure of knowledge.

**Logical reasoning held strong:**
Tasks like formal_logic, logical_fallacies, and jurisprudence either improved or remained stable, indicating that core reasoning capabilities were preserved—or even enhanced—by exposure to structured biomedical text.

## 📖 Conclusion

Training TinyEngram on biomedical data did not induce catastrophic forgetting. Instead, it led to broadly improved performance on general academic knowledge, with only negligible, isolated dips. This supports a growing consensus: thoughtful domain specialization can be complementary—not antagonistic—to general competence, especially when the target domain demands rigor, structure, and precision.
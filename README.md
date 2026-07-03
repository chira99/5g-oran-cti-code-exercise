# LLM-Assisted Cyber Threat Intelligence for 5G O-RAN Security

A complete pipeline for ML-based intrusion detection and LLM-assisted CTI alert generation in 5G Open RAN environments.

## Overview

This project trains an XGBoost classifier on the NetsLab-5GORAN-IDD network dataset to detect five traffic categories (benign + four attack types). Detected threats are converted into structured CTI alerts, enriched by a locally-deployed Qwen2.5-7B LLM via Ollama. As a bonus, SHAP-based feature explanations are added as evidence to ground the LLM assessment (compared against enrichment without SHAP), and the LLM is fine-tuned on a small CTI corpus with LoRA/QLoRA (Unsloth).

```
Network flow → XGBoost classifier → CTI alert (JSON) → Qwen2.5-7B → Incident report
                     ↑                      ↑                ↑
                  SMOTE               SHAP evidence     LoRA fine-tune
```

## Pipeline

| Section | Part | Description |
|---------|------|-------------|
| 0 | Setup | Install dependencies, mount Google Drive, verify GPU |
| 1 | 1 | Load dataset (CSV, with SQLite fallback) |
| 2 | 1 | EDA — null check, class distribution, attack types, categorical features, mutual-information relevance, correlation matrix |
| 3 | 1 | Preprocessing — drop leakage/ID columns, exclude bruteforce, label-encode categoricals, encode target, train/test split, SMOTE + RandomUnderSampler balancing |
| 4 | 1 | Train XGBoost (`multi:softprob`, GPU) + validation loss curve |
| 5 | 1 | Evaluate — accuracy, macro precision/recall/F1, per-class F1, confusion matrix, error analysis |
| 6 | 2 | Structured CTI alert generation — JSON with predicted class, confidence, alternatives, observations, affected component, timestamp |
| 7 | 3 ★ | SHAP TreeExplainer — global/per-class feature importance, SHAP-enhanced CTI alert |
| 8 | 2 | LLM enrichment — Qwen2.5-7B via Ollama, zero-shot with threat knowledge base |
| 9 | 2 | LLM output evaluation — hallucination check, SHAP vs no-SHAP comparison |
| 10 | 2 ★ | LLM fine-tuning (LoRA/QLoRA via Unsloth), base vs fine-tuned comparison, final incident report |
| 11 | 4 | End-to-end demo |

★ Bonus / extended component

## Setup

### Running on Google Colab (recommended)

1. Upload `Network_Dataset.csv` to Google Drive under `MyDrive/CTI/`.
2. For the fine-tuning section (Section 10), also upload `finetune/cti_training_data_chat.json` to `MyDrive/CTI/` (Section 10.2 falls back to an interactive upload if it is missing).
3. Open `notebooks/5g_oran_cti_pipeline.ipynb` in Colab.
4. Set runtime to **T4 GPU** (Runtime → Change runtime type) — required by XGBoost (`device='cuda'`), Ollama, and Unsloth.
5. Run all cells top to bottom. Section 10 frees the GPU held by Ollama (`ollama stop`) before loading Unsloth to avoid an out-of-memory error on the T4.

### Running locally

```bash
pip install -r requirements.txt

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b

# Launch notebook
jupyter notebook notebooks/5g_oran_cti_pipeline.ipynb
```

Update `CSV_PATH` (and the other Drive paths) in the Setup cell to point to your local copies. Sections 0, 4, 8 and 10 install their own dependencies in-notebook (torch, ollama, unsloth, trl, peft, accelerate, datasets); a CUDA GPU is required for training, Ollama inference, and fine-tuning.

## Dataset

See [`data/README.md`](data/README.md) for download instructions. Not included in this repo due to file size (~200 MB, 1,723,817 rows).

## Model

- **Algorithm:** XGBoost (`multi:softprob`, 300 estimators, `max_depth=6`, `tree_method='hist'`, GPU, ~12s training)
- **Imbalance handling:** SMOTE + RandomUnderSampler — 200k samples per class (1M total balanced training set)
- **Classes:** benign, ddos, dos, probe, web (bruteforce excluded — see notebook Section 3.1)
- **Accuracy:** 96.4% · **Macro F1:** 0.961 (see notebook Section 5 for the exact figures from your run)

## LLM

- **Base model:** Qwen2.5-7B-Instruct (4-bit, via Ollama for enrichment)
- **Prompting:** Zero-shot with a static threat knowledge base (MITRE ATT&CK for 5G, ENISA 5G Threat Landscape), temperature 0.1
- **Grounding:** the alert never contains the true label; the system prompt forbids inventing indicators absent from the alert
- **Fine-tuning (Section 10):** LoRA (rank 16) via Unsloth on a 34-example CTI instruction corpus, 3 epochs; base vs fine-tuned assessments are compared

## Outputs

Written to the configured output folder (`MyDrive/CTI/` on Colab):

| File | Description |
|------|-------------|
| `xgb_model.json` | Trained XGBoost model |
| `class_distribution.png` | Class distribution (bar + pie) |
| `mi_feature_relevance.png` | Mutual-information feature relevance |
| `correlation_matrix.png` | Feature correlation heatmap |
| `xgb_loss_curve.png` | XGBoost validation log-loss curve |
| `confusion_matrix.png` | Confusion matrix (raw counts + row-normalised) |
| `shap_global_beeswarm.png` | Global SHAP summary (beeswarm) |
| `shap_global_importance.png` | Global mean \|SHAP\| feature importance |
| `shap_per_class.png` | Per-class SHAP feature importance |
| `qwen25_7b_cti_lora/` | Saved LoRA fine-tuned adapters |
| `sample_incident_report.json` | Sample LLM-generated incident report (submission artifact) |

## Demo

[Demo video link — to be added]

## References

- NetsLab-5GORAN-IDD dataset (NetsLab, University of Aveiro)
- MITRE ATT&CK for 5G
- ENISA Threat Landscape for 5G Networks
- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP)

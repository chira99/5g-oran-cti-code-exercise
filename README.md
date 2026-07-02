# LLM-Assisted Cyber Threat Intelligence for 5G O-RAN Security

A complete pipeline for ML-based intrusion detection and LLM-assisted CTI alert generation in 5G Open RAN environments.

## Overview

This project trains an XGBoost classifier on the NetsLab-5GORAN-IDD network dataset to detect five traffic categories (benign + four attack types). Detected threats are converted into structured CTI alerts enriched by a locally-deployed Qwen2.5-7B LLM via Ollama. As a bonus, SHAP-based feature explanations are added as evidence to ground the LLM assessment and compared against enrichment without SHAP.

```
Network flow → XGBoost classifier → CTI alert (JSON) → Qwen2.5-7B → Incident report
                     ↑                      ↑
                  SMOTE               SHAP evidence
```

## Pipeline

| Section | Part | Description |
|---------|------|-------------|
| 1 | 1 | Load dataset (CSV / SQLite fallback) |
| 2 | 1 | EDA — class distribution, feature distributions, correlation matrix |
| 3 | 1 | Preprocessing — drop leakage columns, label-encode categoricals, SMOTE+RUS balancing |
| 4 | 1 | Train XGBoost (`multi:softprob`) on Google Colab T4 GPU |
| 5 | 1 | Evaluate — accuracy, macro F1, per-class metrics, confusion matrix |
| 6 | 2 | Structured CTI alert generation — JSON with predicted class, confidence, observations |
| 7 | 2 | LLM enrichment — Qwen2.5-7B via Ollama, zero-shot with threat knowledge base |
| 8 | 2 | LLM evaluation — hallucination check, accuracy, incident report |
| 9 | 3 ★ | SHAP TreeExplainer — global/per-class feature importance, SHAP-enhanced CTI alert, LLM comparison |
| 10 | 4 | End-to-end demo |

★ Bonus section

## Setup

### Running on Google Colab (recommended)

1. Upload `Network_Dataset.csv` to Google Drive under `MyDrive/CTI/`.
2. Open `notebooks/5g_oran_cti_pipeline.ipynb` in Colab.
3. Set runtime to **T4 GPU** (Runtime → Change runtime type).
4. Run all cells top to bottom.

### Running locally

```bash
pip install -r requirements.txt

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b

# Launch notebook
jupyter notebook notebooks/5g_oran_cti_pipeline.ipynb
```

Update `CSV_PATH` in the Setup cell to point to your local copy of `Network_Dataset.csv`.

## Dataset

See [`data/README.md`](data/README.md) for download instructions. Not included in this repo due to file size (~200 MB).

## Model

- **Algorithm:** XGBoost (`multi:softprob`, 300 estimators, T4 GPU, 12.5s training)
- **Imbalance handling:** SMOTE + RandomUnderSampler — 200k samples per class (1M total balanced training set)
- **Classes:** benign, ddos, dos, probe, web (bruteforce excluded — see notebook Section 5b)
- **Macro F1:** 0.9609

## LLM

- **Model:** Qwen2.5-7B (4-bit GGUF via Ollama)
- **Prompting:** Zero-shot with static threat knowledge base (MITRE ATT&CK for 5G, ENISA 5G Threat Landscape)
- **Temperature:** 0.1

## Outputs

| File | Description |
|------|-------------|
| `outputs/xgb_model.json` | Trained XGBoost model |
| `outputs/class_distribution.png` | Class distribution plot |
| `outputs/feature_distributions.png` | Feature distribution plots |
| `outputs/correlation_matrix.png` | Feature correlation heatmap |
| `outputs/confusion_matrix.png` | Model confusion matrix |
| `outputs/shap_global.png` | Global SHAP feature importance |
| `outputs/shap_per_class.png` | Per-class SHAP importance |
| `outputs/incident_report.json` | Sample LLM-generated incident report |

## Demo

[Demo video link — to be added]

## References

- NetsLab-5GORAN-IDD dataset
- MITRE ATT&CK for 5G
- ENISA Threat Landscape for 5G Networks
- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP)

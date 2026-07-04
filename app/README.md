# Web UI Demo (Gradio)

A standalone web interface for the 5G O-RAN CTI pipeline — separate from the training
notebook, for the Part 4 demonstration. It wraps the full flow (ML detection → CTI alert →
SHAP → LLM enrichment → incident report) in a browser UI with a **Base ↔ Fine-tuned**
toggle and a public share link.

## What you need on Google Drive (`MyDrive/CTI/`)
The demo runs against artifacts the training notebook produces **once**:
- `Network_Dataset.csv` — the dataset
- `xgb_model.json` — the trained XGBoost model
- `qwen25_7b_cti_lora/` — the fine-tuned LoRA adapters

You do **not** need a live pipeline session or its in-memory data. The demo rebuilds what it
needs (label encoders + sample test events) straight from the CSV. So run the training
notebook only once — just to create `xgb_model.json` and `qwen25_7b_cti_lora/` above — and
after that the demo is fully independent of it.

**Note**: I have also included the `xgb_model.json` trained XGBoost model file and `qwen25_7b_cti_lora/` fine-tuned LoRA adapters in `app/data` folder. You may copy these into your CTI folder in Google Drive if ever required.

## Files
| File | Purpose |
|------|---------|
| `demo_standalone.ipynb` | **Recommended.** One self-contained Colab notebook: installs deps, builds the demo bundle from the CSV, and launches the UI. No dependency on the training notebook or its kernel. |
| `demo_app.py` | The Gradio app (loads model + adapters + bundle from Drive). |
| `cti_core.py` | Standalone pipeline logic (prompts, alert generation, SHAP, LLM enrichment, report builder), copied from the notebook. |
| `export_demo_bundle.py` | *Optional* — snapshots the demo bundle from a **live** pipeline session (reads `X_test` from memory) instead of rebuilding from the CSV. Only useful if you're already in the pipeline's kernel. |

## Run it (Colab, T4 GPU) — recommended
Open **`demo_standalone.ipynb`** in Colab and run all cells top to bottom. It will:
1. install dependencies and mount Drive,
2. clone this repo (to get `cti_core.py` + `demo_app.py` onto the runtime),
3. set up Ollama and pull the base model,
4. build `demo_bundle.pkl` from the CSV (skipped if it already exists on Drive),
5. launch the app and print a `https://…gradio.live` link.

Open the link to view the UI which demonstrates the entire pipeline.


## Notes
- **GPU memory:** the fine-tuned model (~6 GB) loads at startup; Ollama loads the base model
  (~6 GB) on demand. Tight but usually fine on a 15 GB T4. If the Base toggle OOMs on first
  use, set `OLLAMA_KEEP_ALIVE = '0s'` in `demo_app.py` so Ollama frees VRAM after each call.
- To make the app standalone `cti_core.py` has the necessary notebook's cell copied to it. 

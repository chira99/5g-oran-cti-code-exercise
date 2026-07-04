"""
export_demo_bundle.py — run ONCE inside the Colab session AFTER the pipeline notebook
has run, so that these globals exist in memory:
    X_test, y_test, feature_encoders, CLASS_NAMES

It dumps a small, self-contained bundle to Google Drive that the standalone Gradio demo
(demo_app.py) loads — so the app never needs the 200 MB CSV or a full pipeline re-run.

Usage in a Colab cell (after running the notebook, at minimum Sections 0-5):
    %run app/export_demo_bundle.py
or paste the body into a cell.

The trained model (xgb_model.json) and fine-tuned adapters (qwen25_7b_cti_lora/) are
already saved to Drive by the notebook; only encoders + sample events are exported here.
"""
import os
import joblib
import numpy as np

DRIVE_DIR = '/content/drive/MyDrive/CTI'
OUT_PATH = f'{DRIVE_DIR}/demo_bundle.pkl'
N_PER_CLASS = 5          # sample events per class for the demo dropdown
SEED = 42

# ── Sanity check that the notebook has been run ──────────────────────────────
for name in ['X_test', 'y_test', 'feature_encoders', 'CLASS_NAMES']:
    if name not in globals():
        raise RuntimeError(
            f"'{name}' not found. Run the pipeline notebook (Sections 0-5) first, "
            f"then run this in the same session."
        )

# ── Stratified sample of test events (positional indices into X_test/y_test) ──
rng = np.random.RandomState(SEED)
idx = []
for c in range(len(CLASS_NAMES)):
    pool = np.where(y_test == c)[0]
    if len(pool):
        idx += rng.choice(pool, size=min(N_PER_CLASS, len(pool)), replace=False).tolist()

events = X_test.iloc[idx].copy()
events['_true_label'] = [CLASS_NAMES[int(y_test[i])] for i in idx]
events['_orig_idx'] = idx

# ── Bundle everything the app needs (besides the model + adapters on Drive) ───
bundle = {
    'feature_encoders': feature_encoders,      # dict of fitted LabelEncoders
    'class_names': list(CLASS_NAMES),
    'feature_columns': list(X_test.columns),   # feature order for model + SHAP
    'events': events,                          # sample rows + _true_label + _orig_idx
}

os.makedirs(DRIVE_DIR, exist_ok=True)
joblib.dump(bundle, OUT_PATH)

print(f'Saved demo bundle -> {OUT_PATH}')
print(f'  events: {len(events)}  |  class balance: {events["_true_label"].value_counts().to_dict()}')
print(f'  feature columns: {len(bundle["feature_columns"])}')
print('Next: run app/demo_app.py (in this or a fresh GPU session) to launch the web UI.')

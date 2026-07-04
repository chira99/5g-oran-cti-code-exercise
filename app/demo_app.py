"""
demo_app.py — standalone Gradio web demo for the 5G O-RAN CTI pipeline.

Runs in a Colab GPU session. It loads the trained artifacts from Google Drive and wraps
the full pipeline (ML detection -> CTI alert -> SHAP -> LLM enrichment -> incident report)
in a clickable web UI with a Base <-> Fine-tuned toggle, then exposes a public share link.

Prerequisites (all produced/installed by the pipeline notebook):
  - Drive: xgb_model.json, demo_bundle.pkl (run app/export_demo_bundle.py), qwen25_7b_cti_lora/
  - Installed in the session: xgboost, shap, ollama, unsloth, torch  (notebook Section 0)
  - Ollama binary installed + a served model (notebook Section 8.1)

Run (from the repo root, in a Colab cell):
    !python app/demo_app.py
The public https://...gradio.live link is printed on launch.

Note on GPU memory: the fine-tuned model (~6 GB) loads at startup and Ollama loads the base
model (~6 GB) on demand — tight but usually fine on a 15 GB T4. If you hit an OOM when first
selecting the Base model, set OLLAMA_KEEP_ALIVE = '0s' below so Ollama unloads after each call.
"""
import os
import sys
import json
import time
import tempfile
import subprocess

# make cti_core importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import gradio as gr
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gradio'])
    import gradio as gr

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
import ollama as ollama_client

from cti_core import (generate_cti_alert, enrich_base_ollama, enrich_finetuned,
                      build_incident_report)

# ─── Config ──────────────────────────────────────────────────────────────────
DRIVE = '/content/drive/MyDrive/CTI'
MODEL_PATH = f'{DRIVE}/xgb_model.json'
BUNDLE_PATH = f'{DRIVE}/demo_bundle.pkl'
ADAPTER_PATH = f'{DRIVE}/qwen25_7b_cti_lora'
OLLAMA_MODEL = 'qwen2.5:7b'
OLLAMA_KEEP_ALIVE = '5m'   # set to '0s' to free VRAM after each base-model call

SEV_COLOR = {'LOW': '#2e7d32', 'MEDIUM': '#f9a825', 'HIGH': '#ef6c00', 'CRITICAL': '#c62828'}

CTI_ALERT_NOTE = (
    "ℹ️ *This is the structured alert **fed to the LLM as input**. The `llm_assessment` field is "
    "intentionally **absent here** — it is produced by the LLM (see the **LLM Assessment** tab) "
    "and merged into the **Incident report**.*"
)

# ─── Load ML artifacts ───────────────────────────────────────────────────────
print('Loading XGBoost model + demo bundle...')
model = XGBClassifier()
model.load_model(MODEL_PATH)

bundle = joblib.load(BUNDLE_PATH)
feature_encoders = bundle['feature_encoders']
CLASS_NAMES = bundle['class_names']
FEATURE_COLUMNS = bundle['feature_columns']
events = bundle['events'].reset_index(drop=True)

explainer = shap.TreeExplainer(model)
print(f'  {len(events)} demo events | {len(CLASS_NAMES)} classes')

# ─── Load fine-tuned model (Unsloth) ─────────────────────────────────────────
print('Loading fine-tuned LoRA adapters (Unsloth)...')
from unsloth import FastLanguageModel
model_ft, tokenizer_ft = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH, max_seq_length=2048, dtype=None, load_in_4bit=True,
)
FastLanguageModel.for_inference(model_ft)


# ─── Ensure the Ollama base model is available ───────────────────────────────
def ensure_ollama():
    try:
        ollama_client.list()
    except Exception:
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
    try:
        subprocess.run(['ollama', 'pull', OLLAMA_MODEL], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print('[warn] ollama binary not found — run notebook Section 8.1 first for the Base model.')


ensure_ollama()

# ─── Dropdown choices ────────────────────────────────────────────────────────
CHOICES = [f"[{int(oi)}]  true = {tl}"
           for oi, tl in zip(events['_orig_idx'], events['_true_label'])]
CHOICE_POS = {c: i for i, c in enumerate(CHOICES)}


# ─── Pretty HTML rendering for the assessment ────────────────────────────────
def _badge(text, color):
    return (f"<span style='background:{color};color:#fff;padding:3px 12px;border-radius:12px;"
            f"font-weight:700;font-size:0.82em;letter-spacing:0.4px'>{text}</span>")


def _card(label, value, accent):
    return (f"<div style='background:rgba(255,255,255,0.05);border-left:3px solid {accent};"
            f"padding:10px 14px;border-radius:6px;margin:8px 0'>"
            f"<div style='font-size:0.70em;font-weight:700;letter-spacing:0.7px;"
            f"text-transform:uppercase;opacity:0.55;margin-bottom:4px'>{label}</div>"
            f"<div style='line-height:1.55'>{value}</div></div>")


def _section(title):
    return (f"<div style='font-size:0.74em;font-weight:800;letter-spacing:0.6px;opacity:0.5;"
            f"margin:14px 0 2px'>{title}</div>")


def render_assessment_html(assessment, alert, model_label, elapsed, true_label):
    if 'error' in assessment:
        return (f"<div style='color:#ef5350;padding:12px'><b>LLM error:</b> {assessment['error']}"
                f"<br>Is Ollama running and the model pulled? (notebook Section 8.1)</div>")

    sev = str(assessment.get('severity', 'N/A')).upper()
    hr = assessment.get('human_review_required', None)
    hr_yes = hr in (True, 'true', 'True')
    pred = alert['detection']['predicted_class']
    conf = alert['detection']['confidence']
    match = '✅ match' if pred == true_label else '⚠️ mismatch'

    header = (
        "<div style='display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px'>"
        f"{_badge(sev, SEV_COLOR.get(sev, '#555'))}"
        f"{_badge('Human review: ' + ('Required' if hr_yes else 'Not required'), '#ef6c00' if hr_yes else '#2e7d32')}"
        f"<span style='opacity:0.7;font-size:0.85em'>⏱ {elapsed:.1f}s</span>"
        f"<span style='opacity:0.7;font-size:0.85em'>🤖 {model_label}</span>"
        "</div>"
        "<div style='opacity:0.85;font-size:0.9em;margin-bottom:6px'>"
        f"True label <code>{true_label}</code> <span style='opacity:0.6'>(hidden from LLM)</span> · "
        f"Predicted <code>{pred}</code> (conf {conf:.2f}) · {match}</div>"
    )

    body = (
        _section("🔍 ANALYSIS")
        + _card('Threat contextualisation', assessment.get('threat_contextualisation', ''), '#6366f1')
        + _card('Attack correlation', assessment.get('attack_correlation', ''), '#6366f1')
        + _card('Possible impact', assessment.get('possible_impact', ''), '#6366f1')
        + _section("⚠️ SEVERITY")
        + _card('Justification', assessment.get('severity_justification', ''), SEV_COLOR.get(sev, '#888'))
        + _section("🛡️ RESPONSE")
        + _card('Immediate response', assessment.get('immediate_response', ''), '#0891b2')
        + _card('Longer-term mitigation', assessment.get('longer_term_mitigation', ''), '#0891b2')
        + _section("👤 HUMAN REVIEW")
        + _card('Reason', assessment.get('human_review_reason', ''), '#ef6c00' if hr_yes else '#2e7d32')
    )
    return f"<div style='font-family:system-ui,-apple-system,sans-serif'>{header}{body}</div>"


# ─── Core analyze callback ───────────────────────────────────────────────────
def analyze(choice, model_mode):
    pos = CHOICE_POS[choice]
    row = events.iloc[pos]
    x_row_df = events.iloc[[pos]][FEATURE_COLUMNS]
    x_row = x_row_df.iloc[0]
    true_label = row['_true_label']

    proba = model.predict_proba(x_row_df)[0]
    pred_dict = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}

    alert = generate_cti_alert(
        x_row, proba, feature_encoders, CLASS_NAMES,
        explainer=explainer, x_row_df=x_row_df, include_shap=True, tag=int(row['_orig_idx']),
    )

    try:
        if str(model_mode).startswith('Fine'):
            assessment, elapsed = enrich_finetuned(alert, model_ft, tokenizer_ft)
            model_label = f'{OLLAMA_MODEL} (fine-tuned, LoRA rank=16)'
        else:
            assessment, elapsed = enrich_base_ollama(alert, ollama_client, OLLAMA_MODEL)
            model_label = f'{OLLAMA_MODEL} (base + knowledge base)'
    except Exception as e:
        assessment, elapsed, model_label = {'error': str(e)}, 0.0, str(model_mode)

    report = build_incident_report(alert, assessment, model_label, elapsed)
    shap_df = pd.DataFrame(alert.get('shap_evidence', {}).get('top_features', []))
    assess_html = render_assessment_html(assessment, alert, model_label, elapsed, true_label)

    fpath = os.path.join(tempfile.gettempdir(), f"incident_report_{alert['alert_id']}.json")
    with open(fpath, 'w') as f:
        json.dump(report, f, indent=2)

    return pred_dict, alert, shap_df, assess_html, report, fpath


# ─── UI ──────────────────────────────────────────────────────────────────────
with gr.Blocks(title='5G O-RAN CTI Demo', theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 5G O-RAN CTI Pipeline — Live Demo\n"
        "Select a test event → XGBoost detection → structured CTI alert + SHAP evidence → "
        "LLM assessment (base vs fine-tuned) → incident report."
    )
    with gr.Row():
        with gr.Column(scale=1):
            ev = gr.Dropdown(CHOICES, value=CHOICES[0], label='Test event (true label shown for reference)')
            mode = gr.Radio(['Fine-tuned (LoRA)', 'Base (Ollama + KB)'],
                            value='Fine-tuned (LoRA)', label='LLM for enrichment')
            btn = gr.Button('Analyze', variant='primary')
            pred = gr.Label(label='ML prediction (class confidences)', num_top_classes=5)
        with gr.Column(scale=2):
            with gr.Tab('LLM Assessment'):
                assess = gr.HTML()
            with gr.Tab('CTI Alert (JSON)'):
                gr.Markdown(CTI_ALERT_NOTE)
                alert_json = gr.JSON()
            with gr.Tab('SHAP evidence'):
                shap_tbl = gr.Dataframe(
                    headers=['rank', 'feature', 'feature_value', 'shap_value', 'direction'],
                    label='Top-5 SHAP features for the prediction')
            with gr.Tab('Incident report'):
                gr.Markdown("The final incident report artifact (alert + assessment + metadata).")
                report_json = gr.JSON(label='Incident report')
                report_file = gr.File(label='Download incident report (JSON)')

    btn.click(analyze, [ev, mode], [pred, alert_json, shap_tbl, assess, report_json, report_file])

if __name__ == '__main__':
    demo.launch(share=True)

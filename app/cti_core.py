"""
cti_core.py — standalone CTI pipeline logic for the Gradio demo app.

Copied from the notebook (notebooks/5g_oran_cti_pipeline.ipynb) so the web demo is
self-contained and the notebook stays unmodified. If you change the prompts, alert
schema, or enrichment logic in the notebook, mirror the change here.

Functions take explicit arguments (model outputs, encoders, explainer) instead of
relying on notebook globals, so they can be reused by the app.
"""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ─── O-RAN component mapping (by predicted class) ────────────────────────────
# In production this would come from actual network topology config.
COMPONENT_MAP = {
    'ddos':   'O-RAN Central Unit (O-CU)',
    'dos':    'O-RAN Central Unit (O-CU)',
    'probe':  'O-RAN Near-RT RIC',
    'web':    'O-RAN Application Layer / xApp Interface',
    'benign': 'N/A',
}

# ─── Threat knowledge base (base-model system prompt) ────────────────────────
# Sources: MITRE ATT&CK for 5G; ENISA Threat Landscape for 5G Networks;
#          O-RAN Software Community architecture (docs.o-ran-sc.org).
THREAT_KNOWLEDGE = """
## 5G O-RAN Threat Knowledge Base

### O-RAN Architecture Components (criticality for severity scoring)
Rank by blast radius: Near-RT RIC and SMO (highest — govern many elements) >
O-CU, O-DU (RAN protocol processing) > O-RU, application / xApp interfaces.
- O-RU (Radio Unit): Low-PHY and RF processing at cell sites.
- O-DU (Distributed Unit): High-PHY, MAC, RLC processing.
- O-CU (Central Unit): RRC, SDAP, PDCP processing (includes the CU-CP control-plane function).
- Near-RT RIC: near-real-time RAN Intelligent Controller (E2) — runs xApps. Highest criticality.
- SMO: Service Management and Orchestration (O1) — lifecycle management. Highest criticality.

### Known Attack Types and Behaviours

**DoS (Denial of Service):**
Single-source flooding attacks targeting O-RAN components. Common subtypes:
- SYN Flood: Sends large volumes of TCP SYN packets without completing handshake, exhausting connection tables
- UDP Flood: High-rate UDP packets to random ports, consuming bandwidth and CPU
- ICMP Flood: Ping flood exhausting processing resources
- Slowloris: Sends partial HTTP headers slowly, holding connections open to exhaust server threads
Indicators: High src_packets, near-zero dst_packets (one-directional), short duration_sec, RSTRH/S0 connection_state values

**DDoS (Distributed Denial of Service):**
Multi-source flooding. Same subtypes as DoS but originating from many sources simultaneously.
More difficult to block than single-source DoS. Can overwhelm O-CU bandwidth.
Indicators: Same flow-level signatures as DoS (high packet rate, one-directional traffic); source diversity
is not observable from per-flow features alone and requires aggregation across flows.

**Probe / Reconnaissance:**
Scanning attacks to discover network topology before launching targeted attacks.
- TCP Port Scan: Systematically checks which ports are open on target
- UDP Port Scan: Similar to TCP but using UDP probes
- OS Fingerprinting: Analyses TCP/IP stack responses to determine target OS
Indicators: Low bytes per flow (low src_bytes/dst_bytes), SYN-only or SYN-RST connection_state, no established
connections; the probed destination port appears as dst_port in the SHAP evidence when it drives the prediction
Target: Near-RT RIC and SMO interfaces are common reconnaissance targets

**Web Attacks:**
Application-layer attacks targeting O-RAN web interfaces and xApp APIs.
- HTTP Flood: High-rate legitimate-looking HTTP GET/POST requests to overwhelm web server
- SQL Injection: Malicious SQL in HTTP parameters to extract or corrupt database contents
- XSS (Cross-Site Scripting): Injects scripts into web responses
Indicators: Elevated http_transactions, http_status_error=True, file_transferred=True; GET-heavy request
patterns may surface as is_GET_mthd in the SHAP evidence
"""

# ─── System prompts ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a cybersecurity analyst specialising in 5G Open Radio Access Network (O-RAN) security.
You analyse structured network threat alerts and produce detailed, structured incident assessments.

{THREAT_KNOWLEDGE}

ASSESSMENT RULES:
1. Base your assessment ONLY on information in the provided alert. Do not invent IP addresses, port numbers, timestamps, packet counts, or any other indicators not present in the alert.
2. If evidence is insufficient to make a confident determination, state this explicitly rather than guessing.
3. Your severity rating MUST be justified by specific observations from the alert (quote field names and values).
4. When SHAP evidence is provided, you MUST reference specific features by name when assigning severity and recommending a response.
5. Output ONLY valid JSON — no markdown, no prose outside the JSON object.

SEVERITY GUIDELINES (O-RAN context):
Severity combines three factors: detection confidence, attack impact, and the criticality of the affected O-RAN component. Rank criticality by blast radius: the Near-RT RIC and SMO (RAN intelligent control and service management/orchestration — they govern many elements) are highest; the O-CU and O-DU (RAN protocol processing) are next; the O-RU and application / xApp interfaces are lowest.
- LOW: confidence < 0.55, OR benign traffic, OR low-rate scanning / a single failed connection attempt.
- MEDIUM: a confirmed reconnaissance / probe (scanning is preparatory, not yet damaging), OR any active attack at moderate confidence (0.55-0.84).
- HIGH: a high-confidence (>= 0.85) active attack — DoS/DDoS flooding or web exploitation (e.g. SQL injection, XSS) — on the O-CU, O-DU, or an application / xApp interface.
- CRITICAL: a high-confidence (>= 0.85) active attack on the Near-RT RIC or SMO, OR clear data exfiltration from any component.
Rules:
- Reconnaissance / probe never exceeds MEDIUM on its own.
- Never assign CRITICAL when confidence < 0.85.
- Always set human_review_required = true when confidence < 0.75 or severity is HIGH or CRITICAL."""

# Leaner prompt for the fine-tuned model (no knowledge base — learned from training data).
SYSTEM_PROMPT_FT = """You are a cybersecurity analyst specialising in 5G Open Radio Access Network (O-RAN) security. You analyse structured network threat alerts and produce detailed, structured incident assessments.

ASSESSMENT RULES:
1. Base your assessment ONLY on information in the provided alert. Do not invent indicators absent from the alert.
2. If evidence is insufficient, state this explicitly rather than guessing.
3. Severity MUST be justified by specific field values from the alert.
4. When SHAP evidence is provided, reference specific features by name.
5. Output ONLY valid JSON — no markdown, no prose outside the JSON object.

SEVERITY GUIDELINES (O-RAN context):
Severity combines three factors: detection confidence, attack impact, and the criticality of the affected O-RAN component. Rank criticality by blast radius: the Near-RT RIC and SMO (RAN intelligent control and service management/orchestration — they govern many elements) are highest; the O-CU and O-DU (RAN protocol processing) are next; the O-RU and application / xApp interfaces are lowest.
- LOW: confidence < 0.55, OR benign traffic, OR low-rate scanning / a single failed connection attempt.
- MEDIUM: a confirmed reconnaissance / probe (scanning is preparatory, not yet damaging), OR any active attack at moderate confidence (0.55-0.84).
- HIGH: a high-confidence (>= 0.85) active attack — DoS/DDoS flooding or web exploitation (e.g. SQL injection, XSS) — on the O-CU, O-DU, or an application / xApp interface.
- CRITICAL: a high-confidence (>= 0.85) active attack on the Near-RT RIC or SMO, OR clear data exfiltration from any component.
Rules:
- Reconnaissance / probe never exceeds MEDIUM on its own.
- Never assign CRITICAL when confidence < 0.85.
- Always set human_review_required = true when confidence < 0.75 or severity is HIGH or CRITICAL."""

_RESPONSE_SCHEMA = """{
  "threat_contextualisation": "<describe what is happening based on the alert observations>",
  "attack_correlation": "<correlate with known attack behaviours from the knowledge base>",
  "possible_impact": "<what could happen to the O-RAN network if this attack succeeds>",
  "severity": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "severity_justification": "<cite specific field values from the alert that drove this severity rating>",
  "immediate_response": "<specific actions to take in the next 15 minutes>",
  "longer_term_mitigation": "<architectural or policy changes to prevent recurrence>",
  "human_review_required": <true|false>,
  "human_review_reason": "<why human review is or is not needed>"
}"""

ASSESSMENT_FIELDS = [
    'threat_contextualisation', 'attack_correlation', 'possible_impact',
    'severity', 'severity_justification', 'immediate_response',
    'longer_term_mitigation', 'human_review_required', 'human_review_reason',
]


# ─── SHAP evidence ───────────────────────────────────────────────────────────
def get_shap_evidence(explainer, x_row_df, predicted_class_idx, feature_names, top_n=5):
    """Top-N SHAP features driving the predicted class for a single-row DataFrame."""
    sv_raw = explainer.shap_values(x_row_df)
    if isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        shap_for_pred = sv_raw[0, :, predicted_class_idx]
    else:
        shap_for_pred = sv_raw[predicted_class_idx][0]

    s = pd.Series(shap_for_pred, index=feature_names)
    top = s.abs().nlargest(top_n).index

    evidence = []
    for rank, feat in enumerate(top, start=1):
        val = s[feat]
        evidence.append({
            'rank': rank,
            'feature': feat,
            'feature_value': float(x_row_df[feat].iloc[0]),
            'shap_value': round(float(val), 4),
            'direction': 'supports_prediction' if val > 0 else 'opposes_prediction',
        })
    return evidence


# ─── CTI alert generation ────────────────────────────────────────────────────
def generate_cti_alert(x_row, proba, feature_encoders, class_names,
                       explainer=None, x_row_df=None, include_shap=True,
                       alert_id=None, tag=0):
    """Build a structured CTI alert from one flow's features + class probabilities.

    x_row       : pandas Series of the (encoded) feature row
    proba       : 1-D probability vector over classes
    x_row_df    : single-row DataFrame (required for SHAP)
    """
    pred_idx = int(np.argmax(proba))
    pred_class = class_names[pred_idx]
    confidence = float(proba[pred_idx])

    top3 = np.argsort(proba)[::-1][:3]
    alternatives = [
        {'class': class_names[i], 'confidence': round(float(proba[i]), 4)}
        for i in top3 if i != pred_idx
    ][:2]

    proto_str = feature_encoders['proto'].inverse_transform([int(x_row['proto'])])[0]
    service_str = feature_encoders['service'].inverse_transform([int(x_row['service'])])[0]
    state_str = feature_encoders['conn_state'].inverse_transform([int(x_row['conn_state'])])[0]

    alert = {
        'alert_id': alert_id or f'CTI-{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}-{tag:04d}',
        'schema_version': '1.0',
        'detection': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'predicted_class': pred_class,
            'confidence': round(confidence, 4),
            'alternative_predictions': alternatives,
        },
        'network_observations': {
            'protocol': proto_str,
            'service': service_str,
            'connection_state': state_str,
            'duration_sec': round(float(x_row['duration']), 6),
            'src_bytes': int(x_row['src_bytes']),
            'dst_bytes': int(x_row['dst_bytes']),
            'src_packets': int(x_row['src_pkts']),
            'dst_packets': int(x_row['dst_pkts']),
            'src_ip_bytes': int(x_row['src_ip_bytes']),
            'dst_ip_bytes': int(x_row['dst_ip_bytes']),
            'http_transactions': int(x_row['http_trans_depth']),
            'http_status_error': bool(x_row['http_status_error']),
            'file_transferred': bool(x_row['is_file_transfered']),
        },
        'affected_component': COMPONENT_MAP.get(pred_class, 'Unknown O-RAN Component'),
    }

    if include_shap and explainer is not None and x_row_df is not None:
        ev = get_shap_evidence(explainer, x_row_df, pred_idx, list(x_row_df.columns), top_n=5)
        top_feat = ev[0]
        direction_str = 'supported' if top_feat['direction'] == 'supports_prediction' else 'opposed'
        alert['shap_evidence'] = {
            'top_features': ev,
            'summary': (
                f"The feature '{top_feat['feature']}' (value={top_feat['feature_value']}) "
                f"{direction_str} the prediction most strongly (SHAP={top_feat['shap_value']})."
            ),
        }

    return alert


# ─── LLM enrichment ──────────────────────────────────────────────────────────
def _parse_llm_json(raw_text):
    raw_text = raw_text.strip()
    if raw_text.startswith('```'):
        raw_text = raw_text.split('```')[1]
        if raw_text.startswith('json'):
            raw_text = raw_text[4:]
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {'error': 'JSON parse failed', 'raw_response': raw_text}


def enrich_base_ollama(alert, ollama_client, model_name='qwen2.5:7b'):
    """Enrich via the base Qwen2.5-7B served by Ollama (full knowledge-base prompt)."""
    user_prompt = f"""Analyse the following CTI alert for a 5G O-RAN network and provide a structured assessment.

CTI ALERT:
{json.dumps(alert, indent=2)}

Respond ONLY with a JSON object in this exact format:
{_RESPONSE_SCHEMA}"""

    start = time.time()
    response = ollama_client.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ],
        options={'temperature': 0.1, 'num_predict': 1024},
    )
    elapsed = time.time() - start
    return _parse_llm_json(response['message']['content']), elapsed


def enrich_finetuned(alert, model_ft, tokenizer_ft, max_new_tokens=768):
    """Enrich via the fine-tuned LoRA model (lean prompt). Greedy decoding for reproducible JSON."""
    import torch
    model_ft.generation_config.max_length = None  # avoid max_length vs max_new_tokens warning
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT_FT},
        {'role': 'user', 'content': f"[CTI ALERT]\n{json.dumps(alert, indent=2)}"},
    ]
    prompt = tokenizer_ft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer_ft(prompt, return_tensors='pt').to('cuda')

    start = time.time()
    with torch.no_grad():
        out = model_ft.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tokenizer_ft.eos_token_id,
        )
    elapsed = time.time() - start
    raw = tokenizer_ft.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return _parse_llm_json(raw), elapsed


# ─── Incident report ─────────────────────────────────────────────────────────
def build_incident_report(alert, assessment, model_label, elapsed):
    """Wrap the alert + assessment into the final incident report artifact."""
    shap_used = bool(alert.get('shap_evidence'))
    return {
        'incident_id': alert.get('alert_id', 'UNKNOWN'),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'pipeline_version': '5G O-RAN CTI Pipeline v1.0 (web demo)',
        'model_used': model_label,
        'prompting_strategy': 'zero-shot' + (' + SHAP evidence' if shap_used else ''),
        'shap_evidence_used': shap_used,
        'generation_time_s': round(elapsed, 2),
        'cti_alert': {**alert, 'llm_assessment': assessment},
    }

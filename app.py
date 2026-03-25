import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PTL Fault Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;800&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0c10;
    color: #e0e6f0;
    font-family: 'Barlow', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-image:
        linear-gradient(rgba(0,180,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Hero header */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.3em;
    color: #00b4ff;
    text-transform: uppercase;
    margin-bottom: 1rem;
    opacity: 0.8;
}
.hero-title {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 0%, #00b4ff 60%, #0066ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
}
.hero-sub {
    font-size: 1rem;
    font-weight: 300;
    color: #6b7fa3;
    letter-spacing: 0.05em;
}
.hero-line {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #00b4ff, #0066ff);
    margin: 1.5rem auto 0;
    border-radius: 2px;
}

/* Upload zone */
.upload-zone {
    border: 1.5px dashed rgba(0,180,255,0.3);
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    background: rgba(0,180,255,0.03);
    transition: border-color 0.3s;
    margin-bottom: 1.5rem;
}
.upload-zone:hover { border-color: rgba(0,180,255,0.6); }

/* Stat cards */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.8rem;
    font-weight: 400;
    color: #00b4ff;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5a7a;
}

/* Detection box */
.detection-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #00b4ff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.det-class {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #c8d8f0;
}
.det-conf {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #00b4ff;
}

/* Status badges */
.badge-ok    { color: #00e676; }
.badge-nok   { color: #ff5252; }
.badge-aok   { color: #ffd740; }
.badge-unk   { color: #90a4ae; }
.badge-nest  { color: #ff6d00; }

/* Section label */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #4a5a7a;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* Confidence threshold slider */
[data-testid="stSlider"] > div { padding: 0; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0066ff, #00b4ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Barlow', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Image display */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Progress / spinner */
.stSpinner { color: #00b4ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    'baliser_ok':      '#00e676',
    'baliser_aok':     '#ffd740',
    'baliser_nok':     '#ff5252',
    'insulator_ok':    '#00e676',
    'insulator_unk':   '#90a4ae',
    'insulator_nok':   '#ff5252',
    'insulator-nok':   '#ff5252',
    'bird_nest':       '#ff6d00',
    'stockbridge_ok':  '#00e676',
    'stockbridge_nok': '#ff5252',
    'spacer_ok':       '#00e676',
    'spacer_nok':      '#ff5252',
}

STATUS_BADGE = {
    '_ok':   ('OK',      'badge-ok'),
    '_aok':  ('WARN',    'badge-aok'),
    '_nok':  ('-NOK',    'badge-nok'),
    '-nok':  ('-NOK',    'badge-nok'),
    '_unk':  ('UNKNOWN', 'badge-unk'),
    'bird_': ('ALERT',   'badge-nest'),
}

def get_badge(class_name):
    for key, (label, css) in STATUS_BADGE.items():
        if key in class_name:
            return label, css
    return 'INFO', 'badge-unk'

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">// Drone Inspection System</div>
    <div class="hero-title">PTL Fault Detection</div>
    <div class="hero-sub">Power Transmission Line · AI-Powered Analysis</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<div class="section-label">// Model Configuration</div>', unsafe_allow_html=True)

    model_path = st.text_input(
        "Model path",
        value="ptl_fault_detection_best.pt",
        label_visibility="collapsed",
        placeholder="Path to .pt model file...",
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.95,
        value=0.25,
        step=0.05,
        format="%.2f",
    )

    st.markdown('<div class="section-label" style="margin-top:1.5rem">// Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload drone image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    run_btn = st.button("⚡  Run Detection", use_container_width=True)

with col_right:
    st.markdown('<div class="section-label">// Detection Output</div>', unsafe_allow_html=True)

    output_placeholder = st.empty()
    stats_placeholder  = st.empty()
    dets_placeholder   = st.empty()

    if not uploaded_file:
        output_placeholder.markdown("""
        <div style="
            height: 340px;
            border: 1.5px dashed rgba(0,180,255,0.15);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #2a3a5a;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            letter-spacing: 0.15em;
        ">AWAITING IMAGE INPUT</div>
        """, unsafe_allow_html=True)

# ── Show uploaded image preview ───────────────────────────────────────────────
if uploaded_file and not run_btn:
    with col_right:
        output_placeholder.image(uploaded_file, use_container_width=True)

# ── Run detection ─────────────────────────────────────────────────────────────
if run_btn and uploaded_file:
    if not os.path.exists(model_path):
        st.error(f"Model not found at: `{model_path}` — check the path and try again.")
    else:
        with st.spinner("Running inference..."):
            # Load model
            model = load_model(model_path)

            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Run inference
            t0 = time.time()
            results = model.predict(
                source=tmp_path,
                conf=conf_threshold,
                save=False,
                verbose=False,
            )
            elapsed = time.time() - t0

            result = results[0]
            annotated = result.plot()  # BGR numpy array
            annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB

            # Clean up temp file
            os.unlink(tmp_path)

        # ── Show annotated image ──────────────────────────────────────────
        with col_right:
            output_placeholder.image(annotated_rgb, use_container_width=True)

            # ── Stats row ─────────────────────────────────────────────────
            boxes = result.boxes
            n_det = len(boxes)
            classes_detected = []
            if n_det > 0:
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confs     = boxes.conf.cpu().numpy()
                names     = result.names
                classes_detected = [names[i] for i in class_ids]
                avg_conf  = float(np.mean(confs))
                faults    = sum(1 for c in classes_detected if '_nok' in c or '-nok' in c or 'bird' in c)
            else:
                avg_conf = 0.0
                faults   = 0

            stats_placeholder.markdown(f"""
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-value">{n_det}</div>
                    <div class="stat-label">Detections</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color:{'#ff5252' if faults > 0 else '#00e676'}">{faults}</div>
                    <div class="stat-label">Faults</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_conf:.0%}</div>
                    <div class="stat-label">Avg Conf</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{elapsed:.1f}s</div>
                    <div class="stat-label">Inference</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Detection list ────────────────────────────────────────────
            if n_det == 0:
                dets_placeholder.markdown("""
                <div style="text-align:center; padding:2rem; color:#4a5a7a;
                            font-family:'Share Tech Mono',monospace; font-size:0.8rem;">
                    NO FAULTS DETECTED
                </div>
                """, unsafe_allow_html=True)
            else:
                det_html = '<div class="section-label" style="margin-top:1rem">// Detected Components</div>'
                for cls_name, conf in zip(classes_detected, confs):
                    badge_label, badge_css = get_badge(cls_name)
                    color = CLASS_COLORS.get(cls_name, '#00b4ff')
                    det_html += f"""
                    <div class="detection-card" style="border-left-color:{color}">
                        <span class="det-class">{cls_name.replace('_', ' ').upper()}</span>
                        <span>
                            <span class="{badge_css}" style="font-size:0.7rem;
                                font-family:'Share Tech Mono',monospace;
                                margin-right:0.8rem;">{badge_label}</span>
                            <span class="det-conf">{conf:.0%}</span>
                        </span>
                    </div>
                    """
                dets_placeholder.markdown(det_html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 0 1rem;
            font-family:'Share Tech Mono',monospace;
            font-size:0.65rem; letter-spacing:0.2em; color:#1e2a3a;">
    PTL FAULT DETECTION SYSTEM · YOLOV8 · FURNAS DATASET
</div>
""", unsafe_allow_html=True)

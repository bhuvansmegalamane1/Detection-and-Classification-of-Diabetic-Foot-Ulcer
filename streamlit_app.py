import streamlit as st
import numpy as np
from PIL import Image
import torch
import os
import tempfile
import time
import json
import io
import cv2
import plotly.graph_objects as go
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import base64
from ultralytics import YOLO
from predict_with_staging import UlcerStageClassifier, predict_with_staging_instance

# Page configuration
st.set_page_config(
    page_title="Hospital AI Dashboard | DFU Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except: return ""

# --- BRANDING ASSETS ---
LOCAL_LOGO = "logo.png"
FALLBACK_LOGO = "https://cdn-icons-png.flaticon.com/512/2864/2864274.png"
L_B64 = get_base64_of_bin_file(LOCAL_LOGO) if os.path.exists(LOCAL_LOGO) else ""
logo_src = f"data:image/png;base64,{L_B64}" if L_B64 else FALLBACK_LOGO

# --- CSS STYLING & PREMIUM THEME ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* GLOBAL RESET */
    .stApp {{ background-color: #f8fafc; font-family: 'Inter', sans-serif; }}
    
    /* CENTRED WATERMARK */
    [data-testid="stAppViewContainer"]::before {{
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: url("{logo_src}"); background-repeat: no-repeat;
        background-position: center; background-size: 300px; opacity: 0.04;
        z-index: -1; pointer-events: none;
    }}

    .section-card {{
        border-radius: 12px; padding: 1.5rem; background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem;
        border: 1px solid #e2e8f0; color: #1e293b;
    }}
    .status-banner {{
        background: #1e3a8a; color: white; padding: 12px 20px; 
        border-radius: 8px; margin-bottom: 1.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# --- HOSPITAL UTILITIES ---
def check_image_quality(pil_img, threshold=50.0):
    cv_img = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance, (variance < threshold)

def render_trend_analysis(target_pid="Global-01"):
    path = "case_history.json"
    if not os.path.exists(path): return None
    with open(path, 'r') as f:
        try:
            history = json.load(f)
            if not history: return None
            valid_h = []
            for e in history:
                if e.get('pid') != target_pid: continue
                try: 
                    e['dt'] = datetime.strptime(e['date'], "%Y-%m-%d %H:%M")
                    valid_h.append(e)
                except: continue
            
            if not valid_h: return None
            valid_h.sort(key=lambda x: x['dt'])
            subset = valid_h[-8:]
            dates = [e['dt'] for e in subset]
            risks = [e.get('risk', e.get('percent', 0)) for e in subset]
            
            fig = go.Figure()
            # Add Safety Zones
            fig.add_hrect(y0=0, y1=30, fillcolor="#f0fdf4", opacity=0.5, layer="below", line_width=0)
            fig.add_hrect(y0=60, y1=100, fillcolor="#fef2f2", opacity=0.5, layer="below", line_width=0)
            
            mode = 'markers+lines' if len(dates) > 1 else 'markers'
            fig.add_trace(go.Scatter(
                x=dates, y=risks, mode=mode, 
                line=dict(color='#1e3a8a', width=3),
                marker=dict(size=10, color='#1e3a8a', symbol='diamond'),
                name="Risk Index"
            ))
            
            fig.update_layout(
                height=260, 
                title=f"Progression Analysis: {target_pid}",
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(range=[0, 100], ticksuffix="%"),
                showlegend=False
            )
            return fig
        except: return None

def generate_ai_care_plan(data):
    s = data['stage']
    risk = data['risk_score']
    
    # Context-aware clinical logic
    if s == 0:
        p = ["**Prevention**: Daily foot inspection and moisturizing.", "**Footwear**: Therapeutic shoes with pressure relief."]
        n = "No active ulcer. Focus is on preventing skin breakdown."
    elif s == 1:
        p = ["**Wound Care**: Saline cleansing and transparent film dressing.", "**Unloading**: Standard offloading boots."]
        n = "Superficial ulcer detected. High potential for rapid healing with glucose control."
    elif s == 2:
        p = ["**Management**: Surgical debridement of callus/necrotic tissue.", "**Antibiotics**: Assess for sub-clinical infection."]
        n = "Deep ulcer identified. Requires professional debridement to reach healthy tissue."
    elif s == 3:
        p = ["**Urgent**: Immediate culture and sensitivity testing.", "**Vascular**: Doppler study to assess arterial supply."]
        n = "Deep infection with potential osteomyelitis. Hospital-grade monitoring required."
    else:
        p = ["**Emergency**: Immediate surgical consultation for drainage/debridement.", "**Inpatient**: Intravenous antibiotics and metabolic stabilization."]
        n = "Critical gangrenous state. High risk of amputation without emergency intervention."
        
    p.append(f"**Glycemic Target**: Current HbA1c {data['hba1c']}% (Clinical Target: <7.0%).")
    return "\n".join([f"- {i}" for i in p]), n

def draw_risk_gauge(risk_score, level, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = risk_score,
        title = {'text': f"RISK LEVEL: {level}", 'font': {'size': 20, 'color': color, 'weight': 'bold'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': '#f0fdf4'},
                {'range': [30, 60], 'color': '#fffbeb'},
                {'range': [60, 100], 'color': '#fef2f2'}
            ],
        }
    ))
    fig.update_layout(
        height=280, 
        margin=dict(l=25, r=25, t=80, b=20),
        font={'color': "#1e293b", 'family': "Arial"}
    )
    return fig

def calculate_risk_score(si, hba, pul, neu):
    """Hybrid AI/Clinical Risk Engine"""
    base = [10, 35, 65, 95][min(si, 3)]
    
    # Clinical Multipliers
    if hba > 8.5: base += 8           # Delayed healing due to hyperglycemia
    if not pul: base += 12            # Vascular Ischemia risk
    if neu: base += 7                 # Loss of sensation risk
    
    risk = min(base, 99)
    level, color = ("LOW", "#10b981") if risk < 30 else ("MODERATE", "#f59e0b") if risk < 60 else ("HIGH", "#ef4444") if risk < 85 else ("CRITICAL", "#7f1d1d")
    return risk, level, color

def calculate_healing_prognosis(s_idx, risk, hba1c):
    w = [3, 8, 14, 26][min(s_idx, 3)]
    return f"~{w}-{w+4} Weeks"

def load_patient_directory():
    path = "case_history.json"
    if not os.path.exists(path): return ["New Patient"]
    with open(path, 'r') as f:
        try:
            h = json.load(f)
            pids = sorted(list(set([e.get('pid', 'Global-01') for e in h])))
            return ["New Patient"] + pids
        except: return ["New Patient"]

def save_to_history(data):
    f = "case_history.json"
    h = []
    if os.path.exists(f):
        with open(f, 'r') as file:
            try: h = json.load(file)
            except: h = []
    h.append(data)
    with open(f, 'w') as file: json.dump(h, file)

def generate_pdf_report(res, clin, risk_data):
    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    # --- WATERMARK ---
    if os.path.exists("logo.png"):
        try:
            p.saveState(); p.setFillAlpha(0.03)
            p.drawImage("logo.png", w/2-80, h/2-80, width=160, height=160, mask='auto')
            p.restoreState()
        except: pass

    # ═══════════════════════════════════════════
    #  HEADER
    # ═══════════════════════════════════════════
    if os.path.exists("logo.png"):
        try: p.drawImage("logo.png", 45, h-70, width=40, height=40, mask='auto')
        except: pass

    p.setFont("Helvetica-Bold", 16); p.setFillColorRGB(0.12, 0.23, 0.54)
    p.drawString(95, h-50, "NATIONAL DIABETIC FOOT CARE CENTRE")
    p.setFont("Helvetica", 8); p.setFillColorRGB(0.45, 0.45, 0.45)
    p.drawString(95, h-62, "Precision AI Clinical Dashboard  |  Automated Diagnostic Triage Report")
    p.setStrokeColorRGB(0.75, 0.75, 0.75); p.line(45, h-75, w-45, h-75)

    y = h - 95

    # ═══════════════════════════════════════════
    #  SECTION 1: DIAGNOSTIC SUMMARY
    # ═══════════════════════════════════════════
    p.setStrokeColorRGB(0.85, 0.85, 0.88); p.setFillColorRGB(0.97, 0.97, 0.99)
    p.roundRect(45, y-55, w-90, 65, 6, fill=1, stroke=1)

    p.setFillColorRGB(0.12, 0.23, 0.54); p.setFont("Helvetica-Bold", 10)
    p.drawString(55, y, "SECTION I — CLINICAL DIAGNOSTIC SUMMARY")

    p.setFillColorRGB(0, 0, 0); p.setFont("Helvetica", 9)
    y -= 18
    p.drawString(60, y, "Case ID:")
    p.setFont("Helvetica-Bold", 9); p.drawString(105, y, f"DFU-{datetime.now().strftime('%Y%m%d-%H%M')}")
    p.setFont("Helvetica", 9); p.drawString(300, y, "Wagner Stage:")
    p.setFont("Helvetica-Bold", 9); p.drawString(370, y, f"{res['overall_severity']}")

    y -= 15
    p.setFont("Helvetica", 9)
    p.drawString(60, y, "Risk Score:")
    p.setFont("Helvetica-Bold", 9); p.drawString(115, y, f"{risk_data[0]}% ({risk_data[1]})")
    p.setFont("Helvetica", 9); p.drawString(300, y, "HbA1c:")
    p.setFont("Helvetica-Bold", 9); p.drawString(340, y, f"{clin['hba1c']}%")

    y -= 35

    # ═══════════════════════════════════════════
    #  SECTION 2: IMAGE EVIDENCE
    # ═══════════════════════════════════════════
    p.setFillColorRGB(0.12, 0.23, 0.54); p.setFont("Helvetica-Bold", 10)
    p.drawString(55, y, "SECTION II — AI COMPUTER VISION EVIDENCE")
    p.setFont("Helvetica", 8); p.setFillColorRGB(0.4, 0.4, 0.4)
    p.drawString(55, y-12, "Detected ulcer region with bounding box overlay from YOLOv8 pipeline")
    y -= 25

    img_h = 250
    img_drawn = False
    if 'processed_image' in res and res['processed_image'] is not None:
        try:
            pil_img = res['processed_image']
            # processed_image is already a PIL Image in RGB from predict_with_staging
            if not isinstance(pil_img, Image.Image):
                pil_img = Image.fromarray(pil_img)
            img_r = ImageReader(pil_img)
            iw, ih = pil_img.size
            # Scale to fit width while maintaining aspect ratio
            display_w = w - 150
            display_h = int(display_w * ih / iw)
            if display_h > img_h:
                display_h = img_h
                display_w = int(display_h * iw / ih)
            x_offset = (w - display_w) / 2
            p.drawImage(img_r, x_offset, y - display_h, width=display_w, height=display_h, mask='auto')
            y -= (display_h + 15)
            img_drawn = True
        except Exception as ex:
            p.setFillColorRGB(0.8, 0, 0); p.setFont("Helvetica", 9)
            p.drawString(60, y - 20, f"[Image rendering error: {str(ex)[:60]}]")
            y -= 40

    if not img_drawn and 'processed_image' not in res:
        p.setFillColorRGB(0.5, 0.5, 0.5); p.setFont("Helvetica-Oblique", 9)
        p.drawString(60, y - 20, "No processed image available for this session.")
        y -= 40

    # ═══════════════════════════════════════════
    #  SECTION 3: AI SYSTEM SPECS
    # ═══════════════════════════════════════════
    p.setFillColorRGB(0.12, 0.23, 0.54); p.setFont("Helvetica-Bold", 10)
    p.drawString(55, y, "SECTION III — AI SYSTEM SPECIFICATIONS")
    y -= 16
    p.setFont("Helvetica", 9); p.setFillColorRGB(0.2, 0.2, 0.2)
    for line in [
        "Detection Engine:  YOLOv8 Institutional Foot Scan Classifier",
        "Staging Model:     EfficientNet-B0 Medical Pre-trained Suite",
        "Risk Fusion:       Multi-modal HbA1c + Pulse + Neuropathy Integration"
    ]:
        p.drawString(70, y, f"\u2022  {line}")
        y -= 13

    # ═══════════════════════════════════════════
    #  SIGNATURE BLOCK
    # ═══════════════════════════════════════════
    p.setStrokeColorRGB(0.7, 0.7, 0.7); p.line(350, 115, w-45, 115)
    p.setFont("Helvetica-Bold", 8); p.setFillColorRGB(0.12, 0.23, 0.54)
    p.drawString(350, 103, "Authorized Reviewing Physician")
    p.setFont("Helvetica", 7); p.setFillColorRGB(0.5, 0.5, 0.5)
    p.drawString(350, 93, f"EMR Hash: {datetime.now().strftime('%Y%m%d%H%M%S')}")

    # ═══════════════════════════════════════════
    #  FOOTER
    # ═══════════════════════════════════════════
    p.setFont("Helvetica-Oblique", 6.5); p.setFillColorRGB(0.55, 0.55, 0.55)
    p.drawCentredString(w/2, 45, "CONFIDENTIAL: AI-generated clinical support report for physician review only. Not a substitute for clinical judgment.")
    p.drawCentredString(w/2, 36, "Hospital Standard Procedures take precedence. National Diabetic Foot Care Centre Clinical Suite v6.2")

    p.showPage(); p.save(); buf.seek(0)
    return buf

@st.cache_resource
def load_models():
    return YOLO("models/best.pt"), UlcerStageClassifier("configs/stage_config.yaml")

# --- PREMIUM CLINICAL HEADER (LOGO COLUMNS) ---
hcol1, hcol2 = st.columns([1, 6])
with hcol1:
    st.image(logo_src, width=50)

with hcol2:
    st.markdown("<h2 style='margin:0; font-size: 24px; color:#1e3a8a;'>National Diabetic Foot Care Centre</h2>", unsafe_allow_html=True)
    st.caption("Advanced Clinical Intelligence Platform | Multi-modal GenAI Suite")

view = st.radio("Route", ["🏥 Diagnostic Triage Hub", "📋 Clinical Patient Registry"], horizontal=True, label_visibility="collapsed")

def render_diagnostic_hub():
    with st.sidebar:
        st.markdown("### 🧬 Physician Command")
        with st.expander("📁 Patient Case Reference", expanded=True):
            p_opt = load_patient_directory()
            sel_p = st.selectbox("Active Profile", options=p_opt)
            pid = st.text_input("ID/Name", "Patient-001") if sel_p == "New Patient" else sel_p

        with st.expander("👤 Demographics", expanded=False):
            c3, c4 = st.columns(2); age = c3.number_input("Age", 18, 100, 55); gen = c4.radio("Gender", ["M", "F"], horizontal=True)
            mods = st.multiselect("Comorbidities", ["HTN", "Dyslipidemia", "CKD", "Obesity"])
        
        with st.expander("🩹 Clinical Vitals", expanded=True):
            up = st.file_uploader("Upload Scan", type=['jpg','png'], label_visibility="collapsed")
            hba = st.number_input("HbA1c (%)", 4.0, 16.0, 7.5, 0.1)
            dur = st.number_input("Duration (yrs)", 0, 50, 10)
            pul = st.radio("Peripheral Pulse", ["Present", "Absent"], horizontal=True) == "Present"
            neu = st.checkbox("Neuropathy Identified")
            
    if up:
        img = Image.open(up)
        if st.button("🚀 EXECUTE MULTI-MODAL PIPELINE", type="primary", use_container_width=True):
            with st.status("🩺 Processing Clinical Data...", expanded=False):
                dm, sm = load_models(); t_s = time.time()
                with tempfile.TemporaryDirectory() as td:
                    tp = os.path.join(td, "i.jpg"); img.convert('RGB').save(tp)
                    res = predict_with_staging_instance(tp, dm, sm)
                    si = res['detections'][0]['stage_idx'] if res['detections'] else 0
                    ri = calculate_risk_score(si, hba, pul, neu)
                    res.update({'risk': ri, 'healing': calculate_healing_prognosis(si, ri[0], hba), 'time': time.time()-t_s, 'meta': {"pid": pid, "age": age, "gender": gen, "comorbidities": mods, "hba1c": hba, "dur": dur}})
                    res['plan'], res['note'] = generate_ai_care_plan({'stage':si, 'risk_score':ri[0], 'hba1c':hba})
                    st.session_state.results = res; st.rerun()

    if 'results' in st.session_state:
        r = st.session_state.results; ri = r['risk']; m = r['meta']
        
        # 🏥 MAIN DASHBOARD GRID [3:2]
        c1, c2 = st.columns([3, 2], gap="large")
        
        with c1:
            st.markdown(f'''<div class="status-banner"><b>TARGET:</b> {m["pid"]} | <b>STAGE:</b> {r["overall_severity"]} | <b>RISK:</b> {ri[0]}%</div>''', unsafe_allow_html=True)
            st.image(r['processed_image'], use_container_width=True, caption="Computer Vision Staging Evidence")
            
            st.markdown('### 🤖 Medical Reasoning')
            tabs = st.tabs(["📝 Care Plan", "💡 Physician Notes"])
            tabs[0].markdown(r['plan']); tabs[1].info(r['note'])
            
            if st.button("💾 SAVE TO PATIENT RECORD", type="primary", use_container_width=True):
                save_to_history({
                    "pid": m['pid'], 
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"), 
                    "risk": ri[0],
                    "stage": r['overall_severity']
                })
                st.toast("✅ Record synchronized safely.")

        with c2:
            st.plotly_chart(draw_risk_gauge(ri[0], ri[1], ri[2]), use_container_width=True)
            
            h_fig = render_trend_analysis(target_pid=m['pid'])
            if h_fig: st.plotly_chart(h_fig, use_container_width=True)
            else: st.info("No historical data for trend analysis.")
            
            st.markdown("### 📂 Case Management")
            pdf = generate_pdf_report(r, {"hba1c":m['hba1c'], "duration":m['dur']}, ri)
            st.download_button("📩 DOWNLOAD CLINICAL REPORT (PDF)", pdf, f"DFU_{m['pid']}.pdf", use_container_width=True)
            st.caption(f"Inference Latency: {r['time']:.2f}s | Multi-modal Hardware Integration Active")
    else:
        st.info("👆 Please upload a patient clinical scan to begin AI triage.")

def render_patient_registry():
    st.markdown('<h2 style="color: #1e3a8a;">👥 Clinical Patient Records Archive</h2>', unsafe_allow_html=True)
    path = "case_history.json"
    if not os.path.exists(path):
        st.info("The patient database is currently empty. Run a triage session to create records.")
        return

    with open(path, 'r') as f:
        try: history = json.load(f)
        except: history = []
    
    # --- FILTERS ---
    c1, c2 = st.columns([2, 1])
    search_q = c1.text_input("🔍 Search by Patient Name / ID", "", help="Enter the name or ID to filter records.")
    date_q = c2.date_input("📅 Filter by Visit Date", None, help="Only show records from this specific date.")
    
    # Process & Filter
    unique_p = {}
    for e in history:
        p = e.get('pid', 'N/A')
        # Apply Filters
        if search_q.lower() not in p.lower(): continue
        if date_q:
            try:
                e_date = datetime.strptime(e['date'], "%Y-%m-%d %H:%M").date()
                if e_date != date_q: continue
            except: continue
        
        c_risk = e.get('risk', e.get('percent', 0))
        if p not in unique_p:
            unique_p[p] = {"Age": e.get('age', 'N/A'), "Gender": e.get('gender', 'N/A'), "Last Visit": e['date'], "Total": 0, "Current Risk": c_risk}
        unique_p[p]['Total'] += 1
        unique_p[p]['Current Risk'] = c_risk

    if not unique_p:
        st.warning("No records found matching your search criteria.")
        return

    # --- REGISTRY TABLE ---
    st.markdown("### 🏥 Matching Case Files")
    # Convert to list for display
    display_data = []
    for pid, data in unique_p.items():
        display_data.append({
            "Patient ID": pid,
            "Age": data['Age'],
            "Last Visit": data['Last Visit'],
            "Risk Index": f"{data['Current Risk']}%",
            "Entries": data['Total']
        })
    
    st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    # --- DETAILED VIEW ---
    st.divider()
    sel_pid = st.selectbox("📖 Open Clinical Case File", options=list(unique_p.keys()), help="Select a patient from the matching results to view their full progression history.")
    
    if sel_pid:
        d = unique_p[sel_pid]
        st.markdown(f'<div class="section-card"><h3>📁 Deep Review: {sel_pid}</h3></div>', unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].metric("Sessions", d['Total'])
        cols[1].metric("Current Risk", f"{d['Current Risk']}%")
        cols[2].metric("Age", d['Age'])
        cols[3].metric("Gender", d['Gender'])
        
        st.subheader("📊 Healing Progression Timeline")
        fig = render_trend_analysis(target_pid=sel_pid)
        if fig: st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Insufficient historical data for a trend analysis.")

    if st.sidebar.button("🔥 Factory Reset DB", use_container_width=True):
        if os.path.exists("case_history.json"): os.remove("case_history.json")
        st.rerun()

# --- NAVIGATION ROUTER ---
if view == "🏥 Diagnostic Triage Hub":
    render_diagnostic_hub()
else:
    render_patient_registry()

st.sidebar.divider()
st.sidebar.caption("Clinical Suite v6.2 | Developed by AI Medical Systems")
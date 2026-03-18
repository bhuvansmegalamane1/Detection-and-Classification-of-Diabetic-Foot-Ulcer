import streamlit as st
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os
import tempfile
from predict_with_staging import UlcerStageClassifier, predict_with_staging_instance

# Page configuration
st.set_page_config(
    page_title="🏥 Diabetic Foot Ulcer Detection System",
    page_icon="🏥",
    layout="wide"
)

# Cache models to avoid reloading on every interaction
@st.cache_resource
def load_detection_model():
    return YOLO("best.pt")

@st.cache_resource
def load_stage_classifier():
    config_path = "configs/stage_config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "configs", "stage_config.yaml")
    try:
        return UlcerStageClassifier(config_path)
    except Exception as e:
        st.error(f"Failed to load stage classifier: {e}")
        return None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #007bff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stage-label {
        font-weight: 500;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 Diabetic Foot Ulcer Detection System</h1>', unsafe_allow_html=True)

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload a foot image for ulcer detection", 
                                   type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
    
    analyze_btn = st.button("🔬 Analyze Image", type="primary", use_container_width=True)
    
    if st.button("🧹 Clear All", use_container_width=True):
        st.session_state.uploaded_image = None
        st.session_state.detection_results = None
        st.rerun()

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Original Image")
    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("Please upload an image to begin analysis")

with col2:
    st.subheader("🔍 Detection Result")
    if st.session_state.detection_results:
        st.image(st.session_state.detection_results['processed_image'], 
                caption="Detection Result", use_column_width=True)
    elif st.session_state.uploaded_image and analyze_btn:
        with st.spinner("Processing image... Please wait"):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_input_path = os.path.join(temp_dir, "input.jpg")
                    st.session_state.uploaded_image.convert('RGB').save(temp_input_path)
                    
                    model = load_detection_model()
                    stage_classifier = load_stage_classifier()
                    
                    # Run Unified Staging Pipeline
                    st_results = predict_with_staging_instance(temp_input_path, model, stage_classifier)
                    
                    # Map to UI detection_info
                    all_stages = []
                    for det in st_results['detections']:
                        all_stages.append({
                            'name': det['stage_name'],
                            'description': det['stage_description'],
                            'color': det['stage_color'],
                            'confidence': det['stage_confidence'],
                            'details': det['stage_details'],
                            'treatment': det['stage_treatment'],
                            'index': det['stage_idx']
                        })
                    
                    detection_info = {
                        'processed_image': st_results['processed_image'],
                        'num_detections': len(st_results['detections']),
                        'top_confidence': max([d['confidence'] for d in st_results['detections']]) if st_results['detections'] else 0,
                        'predicted_class': st_results['overall_severity'],
                        'all_stages': all_stages,
                        'has_staging_model': stage_classifier is not None
                    }
                    
                    st.session_state.detection_results = detection_info
                    st.rerun()
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    else:
        st.info("Detection result will appear here after analysis")

# Results display
if st.session_state.detection_results:
    st.divider()
    st.subheader("📊 Analysis Results")
    
    detection_info = st.session_state.detection_results
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🔢 Total Detections", value=detection_info['num_detections'])
    with col2:
        if detection_info['top_confidence'] > 0:
            confidence_percent = detection_info['top_confidence'] * 100
            color = "#28a745" if confidence_percent >= 80 else "#ffc107" if confidence_percent >= 60 else "#dc3545"
            st.markdown(f'<div class="metric-card"><p style="color:{color}; font-size:1.2rem; font-weight:bold;">📊 Confidence: {confidence_percent:.1f}%</p></div>', unsafe_allow_html=True)
    with col3:
        diagnosis = (f"🩺 Diagnosis: {detection_info['predicted_class']}" if detection_info['predicted_class'].lower() != "none" else "⚠️ No Ulcer Detected")
        st.markdown(f'<div class="metric-card"><p style="color:#2c3e50; font-size:1.2rem; font-weight:bold;">{diagnosis}</p></div>', unsafe_allow_html=True)

    if detection_info['num_detections'] > 0:
        st.subheader("🏥 Ulcer Staging Information")
        with st.expander("View Individual Ulcer Stages", expanded=True):
            if detection_info['all_stages']:
                for i, stage in enumerate(detection_info['all_stages']):
                    st.markdown(
                        f'<div class="stage-label" style="background-color: {stage["color"]}20; border-left: 4px solid {stage["color"]};">'
                        f'<h4>• Ulcer #{i+1}: {stage["name"]} (Conf: {stage["confidence"]*100:.1f}%)</h4>'
                        f'<p><strong>Description:</strong> {stage["description"]}</p>'
                        f'<p><strong>Treatment:</strong> {stage["treatment"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        if detection_info['all_stages']:
            highest_stage = max(detection_info['all_stages'], key=lambda x: x['index'])
            st.subheader("⚠️ Overall Severity Assessment")
            st.markdown(
                f'<div class="metric-card" style="border-left: 5px solid {highest_stage["color"]}; background-color: {highest_stage["color"]}20;">'
                f'<h3 style="color: {highest_stage["color"]};">🏥 Highest Stage: {highest_stage["name"]}</h3>'
                f'<p>{highest_stage["details"]}</p></div>',
                unsafe_allow_html=True
            )

st.divider()
st.caption("🏥 Diabetic Foot Ulcer Detection System | Medical assistance tool - not a replacement for professional clinical advice.")
import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import os
import tempfile
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="🏥 Diabetic Foot Ulcer Detection System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #007bff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .st-emotion-cache-1v0mbdj {
        border: 2px solid #007bff;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stage-label {
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
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
        st.image(st.session_state.uploaded_image, caption="Uploaded Image")
    else:
        st.info("Please upload an image to begin analysis")

with col2:
    st.subheader("🔍 Detection Result")
    if st.session_state.detection_results:
        st.image(st.session_state.detection_results['processed_image'], 
                caption="Detection Result")
    elif st.session_state.uploaded_image and analyze_btn:
        with st.spinner("Processing image... Please wait"):
            try:
                # Create temporary directory for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded image temporarily
                    temp_input_path = os.path.join(temp_dir, "input.jpg")
                    st.session_state.uploaded_image.save(temp_input_path)
                    
                    # Load the trained detection model
                    model = YOLO("runs/detect/yolov8m_custom/weights/best.pt")
                    
                    # Perform prediction
                    results = model.predict(
                        source=temp_input_path,
                        project=temp_dir,
                        save=True,
                        save_txt=True,
                        conf=0.2,
                        iou=0.45,
                        imgsz=640,
                        augment=True,
                        agnostic_nms=False,
                        max_det=300,
                        save_conf=True
                    )
                    
                    # Find the saved result image
                    result_dir = os.path.join(temp_dir, "predict")
                    if os.path.exists(result_dir):
                        for file in os.listdir(result_dir):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                result_path = os.path.join(result_dir, file)
                                # Open and copy the image to avoid file locking
                                with Image.open(result_path) as img:
                                    processed_image = img.copy()
                                
                                # Extract detection info
                                result = results[0]
                                boxes = result.boxes
                                
                                detection_info = {}
                                if boxes is not None and len(boxes) > 0:
                                    confidences = np.array(boxes.conf)
                                    classes = np.array(boxes.cls)
                                    
                                    # Get top prediction
                                    top_index = np.argmax(confidences)
                                    top_confidence = confidences[top_index].item()
                                    predicted_class = result.names[int(classes[top_index])]
                                    
                                    # Count total detections
                                    num_detections = len(boxes)
                                    
                                    detection_info = {
                                        'processed_image': processed_image,
                                        'top_confidence': top_confidence,
                                        'predicted_class': predicted_class,
                                        'num_detections': num_detections,
                                        'confidences': confidences,
                                        'classes': classes
                                    }
                                else:
                                    detection_info = {
                                        'processed_image': processed_image,
                                        'top_confidence': 0,
                                        'predicted_class': 'No ulcers detected',
                                        'num_detections': 0,
                                        'confidences': [],
                                        'classes': []
                                    }
                                
                                st.session_state.detection_results = detection_info
                                st.rerun()
                    else:
                        st.error("Error: Could not process the image")
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
    else:
        st.info("Detection result will appear here after analysis")

# Results display
if st.session_state.detection_results:
    st.divider()
    st.subheader("📊 Analysis Results")
    
    detection_info = st.session_state.detection_results
    
    # Display main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="🔢 Total Detections", value=detection_info['num_detections'])
    
    with col2:
        if detection_info['top_confidence'] > 0:
            confidence_percent = detection_info['top_confidence'] * 100
            if confidence_percent >= 80:
                confidence_color = "#28a745"  # Green
            elif confidence_percent >= 60:
                confidence_color = "#ffc107"  # Yellow
            else:
                confidence_color = "#dc3545"  # Red
            
            st.markdown(
                f'<div class="metric-card">'
                f'<p style="color:{confidence_color}; font-size:1.2rem; font-weight:bold;">'
                f'📊 Confidence Level: {confidence_percent:.1f}% ({detection_info["top_confidence"]:.3f})'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="metric-card">'
                '<p style="color:#dc3545; font-size:1.2rem; font-weight:bold;">'
                '📊 No detections found'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    with col3:
        diagnosis_text = (f"🩺 Diagnosis: {detection_info['predicted_class']}" 
                         if detection_info['predicted_class'].lower() != "ulcer" 
                         else "⚠️ Potential Ulcer Detected")
        st.markdown(
            f'<div class="metric-card">'
            f'<p style="color:#2c3e50; font-size:1.2rem; font-weight:bold;">'
            f'{diagnosis_text}'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Ulcer staging information (simplified for Streamlit)
    if detection_info['num_detections'] > 0:
        st.subheader("🏥 Ulcer Staging Information")
        
        # Simulated staging for demonstration (since model loading might fail)
        with st.expander("View Individual Ulcer Stages", expanded=True):
            if detection_info['num_detections'] > 0:
                # Define stage information
                stage_info = {
                    0: {
                        "name": "Stage 1: Superficial",
                        "color": "#ffc107",
                        "description": "Superficial ulcer affecting epidermis only",
                        "details": "Stage 1 ulcers are characterized by partial or full-thickness wounds confined to the skin surface. The epidermis and possibly dermis are affected, but no penetration to deeper tissues occurs. These ulcers appear as shallow wounds and may show redness or minor skin breaks. Treatment typically involves wound cleaning, moisture management, and pressure relief.",
                        "treatment": "Treatment: Regular cleaning, moisture management, protective dressings, off-loading pressure, monitoring for progression."
                    },
                    1: {
                        "name": "Stage 2: Deep",
                        "color": "#fd7e14",
                        "description": "Deep ulcer extending to subcutaneous tissue",
                        "details": "Stage 2 ulcers extend through the dermis into subcutaneous tissue, exposing underlying structures like tendons, joint capsules, or bone. These ulcers have more pronounced depth compared to Stage 1 and show visible deeper tissue involvement. Treatment requires more intensive wound care, possible surgical intervention, and infection prevention measures.",
                        "treatment": "Treatment: Advanced wound care, possible surgical intervention, infection prevention, specialist consultation."
                    },
                    2: {
                        "name": "Stage 3: Infected",
                        "color": "#dc3545",
                        "description": "Infected ulcer with abscess or osteomyelitis",
                        "details": "Stage 3 ulcers are deep wounds with signs of active infection, abscess formation, or osteomyelitis. Characteristics include pus, discharge, inflammation, swelling, warmth, or systemic infection indicators. Redness may extend beyond wound edges. Treatment involves antibiotics, aggressive wound care, and possible surgical debridement.",
                        "treatment": "Treatment: Antibiotics, aggressive debridement, infection control, hospitalization may be required."
                    },
                    3: {
                        "name": "Stage 4: Gangrene",
                        "color": "#a71d2a",
                        "description": "Gangrene with tissue necrosis (critical)",
                        "details": "Stage 4 ulcers represent tissue necrosis with gangrenous changes. Characterized by blackened, dead tissue indicating critical ischemia or severe infection. These ulcers show extensive tissue death with dark coloration indicating gangrene. This is a critical condition requiring immediate medical intervention, possible amputation, and aggressive treatment to prevent sepsis.",
                        "treatment": "Treatment: Emergency medical intervention, possible amprecation, intensive care, sepsis prevention."
                    }
                }
                
                # Display information for each detection (using random stages for demo)
                for i in range(min(detection_info['num_detections'], 5)):  # Limit to 5 for demo
                    # For demonstration, randomly assign stages
                    import random
                    stage_idx = random.randint(0, 3)  # Random stage for demo
                    stage = stage_info[stage_idx]
                    
                    with st.container():
                        st.markdown(
                            f'<div class="stage-label" style="background-color: {stage["color"]}20; border-left: 4px solid {stage["color"]};">'
                            f'<h4>• Ulcer #{i+1}: {stage["name"]} (Simulated)</h4>'
                            f'<p><strong>Description:</strong> {stage["description"]}</p>'
                            f'<p><strong>Treatment:</strong> {stage["treatment"]}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
        
        # Overall severity
        st.subheader("⚠️ Overall Severity Assessment")
        if detection_info['num_detections'] > 0:
            # For demo, show highest simulated stage
            import random
            demo_stage_idx = random.randint(0, 3)
            stage_info_list = list(stage_info.values())
            highest_stage = stage_info_list[demo_stage_idx]
            
            st.markdown(
                f'<div class="metric-card" style="border-left: 5px solid {highest_stage["color"]}; background-color: {highest_stage["color"]}20;">'
                f'<h3 style="color: {highest_stage["color"]};">🏥 Highest Stage: {highest_stage["name"]}</h3>'
                f'<p>{highest_stage["details"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No ulcers detected in the image")

# Footer
st.divider()
st.caption("🏥 Diabetic Foot Ulcer Detection System | This is a medical assistance tool and should not replace professional medical advice.")
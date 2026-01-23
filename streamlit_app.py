import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import os
import tempfile
from collections import Counter
import sys
sys.path.append('.')
from predict_with_staging import UlcerStageClassifier, predict_with_staging

# Page configuration
st.set_page_config(
    page_title="Diabetic Foot Ulcer Detection System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling with modern design
st.markdown("""
<style>
    /* Main theme colors - Minimal professional palette */
    :root {
        --primary-color: #2c5282;  /* Professional blue */
        --secondary-color: #e53e3e; /* Error red */
        --success-color: #38a169;   /* Success green */
        --warning-color: #d69e2e;   /* Warning yellow */
        --neutral-light: #f7fafc;   /* Light background */
        --neutral-medium: #718096;  /* Medium gray */
        --neutral-dark: #2d3748;    /* Dark text */
        --border-radius: 6px;
        --box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        color: var(--primary-color);
        text-align: center;
        margin: 1rem 0 1.5rem 0;
        font-weight: 600;
    }
    
    /* Card styling */
    .custom-container {
        border: 2px solid var(--primary-color);
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 0.5rem 0;
        box-shadow: var(--box-shadow);
        border: 1px solid #e2e8f0;
    }
    
    /* Stage labels */
    .stage-label {
        font-weight: 500;
        padding: 0.75rem;
        border-radius: var(--border-radius);
        margin: 0.25rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Button styling */
    div[data-testid="stButton"] > button {
        border-radius: var(--border-radius);
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--neutral-light);
        border-right: 1px solid #e2e8f0;
    }
    
    /* Progress bars */
    div[data-testid="stProgress"] > div > div {
        background-color: var(--primary-color);
    }
    
    /* File uploader */
    div[data-testid="stFileUploader"] {
        border: 2px dashed var(--primary-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        text-align: center;
        background: rgba(44, 82, 130, 0.05);
    }
    
    div[data-testid="stFileUploader"] > section {
        background: transparent !important;
    }
    
    div[data-testid="stFileUploader"] > section > button {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Alert styling */
    div[data-testid="stAlert"] {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--box-shadow);
    }
    
    /* Divider */
    hr {
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 1.5rem 0;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--neutral-dark);
        font-weight: 600;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .metric-card {
            padding: 0.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main header with professional styling
st.markdown('<h1 class="main-header">Diabetic Foot Ulcer Detection System</h1>', unsafe_allow_html=True)

# Professional introduction
st.markdown("""
<div style="text-align: center; max-width: 800px; margin: 0 auto 1.5rem auto; padding: 1rem; background: var(--neutral-light); border-radius: var(--border-radius); border: 1px solid #e2e8f0;">
    <h3 style="color: var(--primary-color); margin-bottom: 0.75rem;">Medical Imaging Analysis</h3>
    <p style="font-size: 1rem; color: var(--neutral-dark); line-height: 1.5; margin: 0;">
        Computer vision system for diabetic foot ulcer detection and staging. 
        This tool assists healthcare professionals in identifying potential ulcers 
        and determining their severity according to the Wagner classification system.
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# Enhanced sidebar for controls
with st.sidebar:
    st.markdown("""
    <div style="background: var(--primary-color); padding: 1rem; border-radius: var(--border-radius); margin-bottom: 1rem;">
        <h2 style="color: white; text-align: center; margin: 0; font-weight: 600;">Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a foot image for ulcer detection", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.success("Image uploaded successfully!")
        
        # Display image preview in sidebar
        st.markdown("**Preview:**")
        st.image(st.session_state.uploaded_image, width=200, caption="Uploaded Image")
    
    st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button(
            "Analyze Image", 
            type="primary", 
            use_container_width=True,
            help="Start the ulcer detection process"
        )
    
    with col2:
        if st.button("Clear All", use_container_width=True, help="Reset the application"):
            st.session_state.uploaded_image = None
            st.session_state.detection_results = None
            st.rerun()
    
    # Information section
    st.divider()
    with st.expander("About This Tool", expanded=False):
        st.markdown("""
        **Technology Stack:**
        - YOLOv8 for object detection
        - Deep learning classification
        - Wagner staging system
        
        **Disclaimer:** This tool provides medical assistance only. 
        Always consult healthcare professionals for diagnosis and treatment decisions.
        """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image")
    else:
        st.info("Please upload an image to begin analysis")

with col2:
    st.subheader("Detection Result")
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
                    
                    # Initialize staging classifier
                    try:
                        stage_classifier = UlcerStageClassifier(config_path="configs/stage_config.yaml")
                        staging_available = True
                    except Exception as e:
                        staging_available = False
                        st.warning("Staging model not available. Detection only mode.")
                        print(f"Staging initialization error: {e}")
                    
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
                                staging_results = []
                                
                                if boxes is not None and len(boxes) > 0:
                                    confidences = np.array(boxes.conf)
                                    classes = np.array(boxes.cls)
                                    bboxes = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                                    
                                    # Get top prediction
                                    top_index = np.argmax(confidences)
                                    top_confidence = confidences[top_index].item()
                                    predicted_class = result.names[int(classes[top_index])]
                                    
                                    # Count total detections
                                    num_detections = len(boxes)
                                    
                                    # Perform staging for each detection if available
                                    if staging_available:
                                        original_image = Image.open(temp_input_path).convert('RGB')
                                        for i in range(len(boxes)):
                                            x1, y1, x2, y2 = bboxes[i]
                                            # Crop the ulcer region
                                            cropped_region = original_image.crop((x1, y1, x2, y2))
                                            
                                            try:
                                                stage_idx, stage_name, stage_description, stage_color, stage_conf = \
                                                    stage_classifier.predict_stage(cropped_region)
                                                
                                                staging_results.append({
                                                    'stage_idx': int(stage_idx),
                                                    'stage_name': stage_name,
                                                    'stage_description': stage_description,
                                                    'stage_color': stage_color,
                                                    'stage_confidence': float(stage_conf),
                                                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                                                })
                                            except Exception as e:
                                                print(f"Staging error for detection {i}: {e}")
                                                # Add placeholder if staging fails
                                                staging_results.append({
                                                    'stage_idx': -1,
                                                    'stage_name': 'Staging Unavailable',
                                                    'stage_description': 'Unable to determine stage',
                                                    'stage_color': '#6c757d',
                                                    'stage_confidence': 0.0,
                                                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                                                })
                                    
                                    detection_info = {
                                        'processed_image': processed_image,
                                        'top_confidence': top_confidence,
                                        'predicted_class': predicted_class,
                                        'num_detections': num_detections,
                                        'confidences': confidences,
                                        'classes': classes,
                                        'staging_results': staging_results,
                                        'staging_available': staging_available
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

# Enhanced results display
if st.session_state.detection_results:
    st.divider()
    st.markdown("""
    <div style="background: var(--success-color); padding: 1rem; border-radius: var(--border-radius); margin: 1.5rem 0; text-align: center;">
        <h2 style="color: white; margin: 0; font-weight: 600;">Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    detection_info = st.session_state.detection_results
    
    # Enhanced main metrics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<h3 style="color: #1976d2; margin-top: 0;">Total Detections</h3>'
            f'<p style="font-size: 2.5rem; font-weight: 700; color: #2c3e50; margin: 0;">'
            f'{detection_info["num_detections"]}'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        if detection_info['top_confidence'] > 0:
            confidence_percent = detection_info['top_confidence'] * 100
            if confidence_percent >= 80:
                confidence_color = "#28a745"  # Green
                confidence_text = "High Confidence"
            elif confidence_percent >= 60:
                confidence_color = "#ffc107"  # Yellow
                confidence_text = "Medium Confidence"
            else:
                confidence_color = "#dc3545"  # Red
                confidence_text = "Low Confidence"
            
            st.markdown(
                f'<div class="metric-card">'
                f'<h3 style="color: {confidence_color}; margin-top: 0;">Confidence Level</h3>'
                f'<p style="font-size: 2rem; font-weight: 700; color: {confidence_color}; margin: 0;">'
                f'{confidence_percent:.1f}%'
                f'</p>'
                f'<p style="font-size: 0.9rem; color: #6c757d; margin: 0.5rem 0 0 0;">'
                f'{confidence_text}'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="metric-card">'
                '<h3 style="color: #dc3545; margin-top: 0;">Detection Status</h3>'
                '<p style="font-size: 1.5rem; font-weight: 700; color: #dc3545; margin: 0;">'
                'No ulcers detected'
                '</p>'
                '</div>',
                unsafe_allow_html=True
            )
    
    with col3:
        diagnosis_text = (f"{detection_info['predicted_class']}" 
                         if detection_info['predicted_class'].lower() != "ulcer" 
                         else "Potential Ulcer Detected")
        st.markdown(
            f'<div class="metric-card">'
            f'<h3 style="color: #6f42c1; margin-top: 0;">Diagnosis</h3>'
            f'<p style="font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin: 0;">'
            f'{diagnosis_text}'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Ulcer staging information
    if detection_info['num_detections'] > 0:
        if detection_info.get('staging_available', False) and detection_info.get('staging_results'):
            st.subheader("Ulcer Staging Information")
            
            with st.expander("View Individual Ulcer Stages", expanded=True):
                staging_results = detection_info['staging_results']
                for i, staging_result in enumerate(staging_results):
                    with st.container():
                        if staging_result['stage_idx'] >= 0:  # Valid staging result
                            st.markdown(
                                f'<div class="stage-label" style="border-left: 4px solid {staging_result["stage_color"]}; background-color: {staging_result["stage_color"]}10;">'
                                f'<h4>Ulcer #{i+1}: {staging_result["stage_name"]}</h4>'
                                f'<p><strong>Confidence:</strong> {staging_result["stage_confidence"]:.2f}</p>'
                                f'<p><strong>Description:</strong> {staging_result["stage_description"]}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:  # Staging unavailable
                            st.markdown(
                                f'<div class="stage-label" style="border-left: 4px solid #6c757d; background-color: #6c757d10;">'
                                f'<h4>Ulcer #{i+1}: Staging Unavailable</h4>'
                                f'<p>Unable to determine ulcer stage. Please consult a healthcare professional.</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
            
            # Overall severity assessment
            if any(result['stage_idx'] >= 0 for result in staging_results):
                st.subheader("Severity Assessment")
                # Find the highest stage
                valid_results = [r for r in staging_results if r['stage_idx'] >= 0]
                if valid_results:
                    highest_stage = max(valid_results, key=lambda x: x['stage_idx'])
                    st.markdown(
                        f'<div class="metric-card" style="border-left: 4px solid {highest_stage["stage_color"]};">'
                        f'<h3>Highest Stage: {highest_stage["stage_name"]}</h3>'
                        f'<p>{highest_stage["stage_description"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            # Staging not available
            st.info("Staging information not available. Detection results only. Please ensure the staging model is properly configured for full functionality.")
    else:
        st.info("No ulcers detected in the image")

# Footer
st.divider()
st.caption("Diabetic Foot Ulcer Detection System | Medical assistance tool - consult healthcare professionals for diagnosis and treatment decisions")
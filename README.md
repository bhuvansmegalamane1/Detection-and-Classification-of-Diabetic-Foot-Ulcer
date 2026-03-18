# Diabetic Foot Ulcer Detection System

🏥 A machine learning-powered system for detecting and staging diabetic foot ulcers using computer vision.

## 🎯 Overview

This project implements a two-stage approach for diabetic foot ulcer analysis:
- **Stage 1**: YOLOv8 object detection to locate ulcer regions
- **Stage 2**: CNN-based classification to determine ulcer severity (Stages 1-4)

## 🚀 Features

- 🔍 **Accurate Detection**: Identifies ulcer locations in foot images
- 🏥 **Severity Staging**: Classifies ulcers into 4 clinical stages based on Wagner criteria
- 💡 **Medical Guidance**: Provides treatment recommendations for each stage
- 📊 **Multi-Ulcer Support**: Handles multiple ulcers in a single image
- 🌐 **Web Interface**: Streamlit-based web application for easy access

## 📁 Repository Structure

```
Foot_Ulcer/
├── streamlit_app.py      # Web application interface
├── train.py             # Model training scripts
├── requirements.txt     # Python dependencies
├── stage_config.yaml    # Configuration files
└── README.md           # This file
```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Foot_Ulcer.git
cd Foot_Ulcer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models (not included in repo):
   - YOLOv8 detection model
   - Stage classification model

## 🖥️ Usage

### Web Application (Recommended)
```bash
streamlit run streamlit_app.py
```

> **Note**: The legacy Desktop GUI (`GUI.py`) has been deprecated and removed. All features are now consolidated into the more robust Streamlit web interface.

## 🏗️ Model Architecture

- **Detection**: YOLOv8m pre-trained model fine-tuned on ulcer dataset
- **Classification**: EfficientNet-B0 architecture for stage determination
- **Training**: Custom dataset with Wagner staging annotations

## 📈 Performance

| Metric | Score |
|--------|-------|
| Detection Accuracy | ~85% |
| Stage Classification | ~78% |
| Processing Time | <2 seconds per image |

## 🏥 Clinical Staging

The system classifies ulcers into 4 stages based on Wagner criteria:

**Stage 1**: Superficial ulcers affecting skin only  
**Stage 2**: Deep ulcers extending to subcutaneous tissue  
**Stage 3**: Infected ulcers with abscess or osteomyelitis  
**Stage 4**: Gangrene with tissue necrosis (critical)

## ⚠️ Important Notice

This system provides medical assistance and educational purposes only. It should NOT replace professional medical diagnosis. Always consult healthcare professionals for proper medical care.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📄 License

This project is for educational and research purposes.

---
*Developed for diabetic foot ulcer detection and staging*
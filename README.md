# AI-Powered Diabetic Foot Ulcer Multi-Modal GenAI Clinical Platform

A comprehensive, hospital-grade AI solution for early detection, classification, and multi-modal risk assessment of diabetic foot ulcers (DFU). This platform integrates computer vision, clinical parameter analysis, and generative AI to provide actionable medical insights.

---

## 🏥 Project Overview
Diabetic foot ulcers are a leading cause of non-traumatic lower-limb amputations worldwide. Our platform provides a multi-step clinical pipeline for automated diagnosis and patient management:
1. **Detection**: Identifying ulcer locations using YOLOv8.
2. **Classification**: Grading severity (Wagner stages) via EfficientNet.
3. **Risk Scoring**: Integrating HbA1c, distal pulse, and neuropathy data.
4. **Clinical Logic**: Generating GenAI-driven medical reasoning.
5. **PDF Reporting**: Delivering professional clinical reports for practitioners.

---

## 🔥 Key Features
- **YOLOv8 Ulcer Detection**: Real-time identification of ulcer regions with high precision.
- **EfficientNet Staging**: Deep learning-based classification into Wagner Stages (0-5).
- **Multi-modal Risk Engine**: Correlates demographic data and clinical vitals (HbA1c, Pulse) for individualized risk assessment.
- **GenAI reasoning**: Leverages large language models to explain AI outcomes and recommend clinical next steps.
- **Clinical Dashboard**: A premium Streamlit-based UI designed for hospital settings.
- **PDF Clinical Report**: Automated generation of doctor-grade reports with detected images and evidence.
- **Trend Analysis**: Tracks historical patient data for longitudinal ulcer progression monitoring.
- **Image Quality Check**: Pre-processing step to filter blurry or non-relevant images.

---

## 🏗️ Clinical Pipeline
The system follows a sequential multi-stage analysis:
`User` ➔ `Streamlit UI` ➔ `YOLO Detection` ➔ `EfficientNet Staging` ➔ `Risk Engine` ➔ `GenAI Reasoning` ➔ `PDF Report` ➔ `Patient History`

---

## 📸 Screenshots
The system includes:
- **`screenshots/ui.png`**: The main clinical dashboard.
- **`screenshots/detection.png`**: YOLOv8 ulcer detection in action.
- **`screenshots/report.png`**: An example of the generated PDF clinical report.

---

## 🚀 Installation & Usage

### ⚙️ Prerequisites
- Python 3.9+
- GPU recommended (for inference performance)

### 📦 Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/bhuvansmegalamane1/Detection-and-Classification-of-Diabetic-Foot-Ulcer.git
   cd Detection-and-Classification-of-Diabetic-Foot-Ulcer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🧰 Tech Stack
- **Framework**: Streamlit (UI/Dashboard)
- **Computer Vision**: YOLOv8, EfficientNet, OpenCV
- **Deep Learning**: PyTorch, Ultralytics
- **Reasoning Engine**: Google Gemini API (GenAI)
- **Reporting**: ReportLab (PDF Generation)
- **Data Visualization**: Plotly, Matplotlib, Seaborn
- **Data Persistence**: JSON/CSV Local Database

---

## 🔬 Scientific Methodology
The platform follows international clinical guidelines (IWGDF) for ulcer classification and risk assessment, ensuring the AI's recommendations align with real-world medical practice.

---

## ⚖️ Disclaimer
*This tool is intended for clinical demonstration and research purposes only. It is not an FDA-approved medical device and should not replace professional medical judgment.*
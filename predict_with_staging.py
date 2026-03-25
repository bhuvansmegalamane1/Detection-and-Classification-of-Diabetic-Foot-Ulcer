import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import yaml
from torchvision import models


class UlcerStageClassifier:
    def __init__(self, config_path="configs/stage_config.yaml", model=None):
        """
        Initialize the ulcer staging classifier with canonical config
        """
        # Load configuration
        if not os.path.exists(config_path):
             config_path = os.path.join(os.path.dirname(__file__), "configs", "stage_config.yaml")
             
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load the trained stage classification model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None:
            self.model = model
        else:
            self.model = self.load_model()
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.config['stage_classifier']['input_size'], 
                              self.config['stage_classifier']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """
        Load architecture and validate training contract (Comment 2)
        """
        # Initialize the model architecture
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 
                                      self.config['stage_classifier']['num_classes'])
        
        # Load the trained weights
        model_path = self.config['stage_classifier']['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if os.path.getsize(model_path) < 1000:
             raise RuntimeError("Staging model is currently a placeholder and cannot be loaded for inference.")
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            
            # CANONICAL VALIDATION (Comment 2)
            # Ensure the model's trained class mapping matches our clinical config
            if 'canonical_order' in checkpoint:
                config_order = self.config['stage_classifier']['canonical_order']
                if checkpoint['canonical_order'] != config_order:
                    raise ValueError(f"CRITICAL: Model contract mismatch! "
                                   f"Trained on {checkpoint['canonical_order']} "
                                   f"but config expects {config_order}")
        except Exception as e:
             raise RuntimeError(f"Model Integrity Error: {e}")
            
        model = model.to(self.device)
        return model

    def predict_stage(self, cropped_region):
        """
        Predict the stage of a cropped ulcer region
        Args:
            cropped_region: PIL Image of the cropped ulcer region
        Returns:
            tuple: (stage_idx, stage_name, stage_description, stage_color, stage_confidence)
        """
        # Preprocess the image
        input_tensor = self.transform(cropped_region).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            stage_idx = int(predicted.item())
            stage_confidence = float(confidence.item())
            
            # Normalize key access: YAML keys are integers
            # Supporting both int and str keys for robustness if YAML changes
            def get_metadata(config_key, idx):
                data = self.config.get(config_key, {})
                if idx in data:
                    return data[idx]
                if str(idx) in data:
                    return data[str(idx)]
                return "Unknown"

            stage_name = get_metadata('stage_names', stage_idx)
            stage_description = get_metadata('stage_descriptions', stage_idx)
            stage_color = get_metadata('stage_colors', stage_idx)
            stage_detail = get_metadata('stage_detailed_info', stage_idx)
            stage_treatment = get_metadata('stage_treatment_guidance', stage_idx)
            
            # MEDICAL SAFETY GUARD: If confidence is too low, do not return a specific stage.
            # This prevents dangerous false-negatives (like Stage 1 for an infection).
            if stage_confidence < 0.55:
                # Keep the predicted stage_idx but modify name and description for uncertainty
                stage_name = f"⚠️ Review Required | {stage_name}"
                stage_description = f"The automated staging model is not highly confident in this assessment. Human clinical assessment is MANDATORY. Predicted: {stage_description}"
                stage_color = "#dc3545" # Red for warning
                stage_detail = f"This ulcer shows features that do not clearly fit a standard superficial category. It may represent a deep infection or critical ischemia. Do not rely on automated staging for this image. Original details: {stage_detail}"
                stage_treatment = f"Action: Immediate referral to a diabetic foot specialist or wound care clinic for physical assessment. Original guidance: {stage_treatment}"
            elif stage_confidence < 0.75:
                # Decisive but cautious labeling for high-accuracy hackathon demo
                stage_name = f"⚠️ {stage_name} (Review Required)"
        
        return stage_idx, stage_name, stage_description, stage_color, stage_confidence, stage_detail, stage_treatment


def draw_detection_labels(image, detections):
    """
    Draw bounding boxes and stage labels on the image for visual confirmation
    """
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    
    # Try to load a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    for det in detections:
        box = det['box']
        stage_name = det['stage_name']
        color = det['stage_color']
        conf = det['stage_confidence']
        
        # Draw bounding box
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=4)
        
        # Draw label background
        label = f"{stage_name} ({conf*100:.1f}%)"
        # Get label size (textbbox returns (x, y, x2, y2))
        try:
            bbox = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle(bbox, fill=color)
        except:
             draw.rectangle([box[0], box[1]-30, box[0]+200, box[1]], fill=color)
             
        # Draw text
        draw.text((box[0], box[1] if 'bbox' not in locals() else box[1]-20), label, fill="white", font=font)
        
    return image


def predict_with_staging(image_path, detection_model_path="models/best.pt", 
                        config_path="configs/stage_config.yaml"):
    """
    Backward compatibility wrapper that creates new models (avoid using for repeated calls)
    """
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "configs", "stage_config.yaml")
        
    detection_model = YOLO(detection_model_path)
    stage_classifier = UlcerStageClassifier(config_path)
    return predict_with_staging_instance(image_path, detection_model, stage_classifier)


def predict_with_staging_instance(image_path, detection_model, stage_classifier):
    """
    Run two-stage inference using provided model instances
    Args:
        image_path: Path to input image
        detection_model: Loaded YOLO detection model
        stage_classifier: Loaded UlcerStageClassifier instance
    Returns:
        dict: Combined detection and staging results
    """
    # Load the input image
    original_image = Image.open(image_path).convert('RGB')
    img_width, img_height = original_image.size
    
    # Stage 1: Run ulcer detection
    detection_results = detection_model.predict(
        source=image_path,
        conf=0.2,
        iou=0.45,
        imgsz=640,
        augment=True,
        agnostic_nms=False,
        max_det=300,
        save_conf=True
    )
    
    detection_result = detection_results[0]
    boxes = detection_result.boxes
    
    results = {
        'image_path': image_path,
        'detections': [],
        'overall_severity': 'None',
        'raw_boxes': boxes # Keep for name mapping if needed
    }
    
    if boxes is not None and len(boxes) > 0:
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        bboxes = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
        
        # Process each detected ulcer
        for i in range(len(boxes)):
            x1, y1, x2, y2 = bboxes[i]
            
            # NORMALIZATION & BOUNDS CHECK (Comment 5)
            # Clip to image boundaries and convert to integers
            x1 = max(0, int(np.floor(x1)))
            y1 = max(0, int(np.floor(y1)))
            x2 = min(img_width, int(np.ceil(x2)))
            y2 = min(img_height, int(np.ceil(y2)))
            
            # Reject invalid or too-small crops
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                print(f"Skipping too-small crop for detection {i}: ({x1},{y1}) to ({x2},{y2})")
                continue
                
            detection_conf = confidences[i]
            
            # Crop the ulcer region from the original image safely
            cropped_region = original_image.crop((x1, y1, x2, y2))
            
            # Stage 2: Classify the stage of the ulcer
            stage_idx, stage_name, stage_description, stage_color, stage_conf, stage_detail, stage_treat = \
                stage_classifier.predict_stage(cropped_region)
            
            # Store detection and staging results
            detection_info = {
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(detection_conf),
                'stage_idx': int(stage_idx),
                'stage_name': stage_name,
                'stage_description': stage_description,
                'stage_color': stage_color,
                'stage_confidence': float(stage_conf),
                'stage_details': stage_detail,
                'stage_treatment': stage_treat
            }
            results['detections'].append(detection_info)
        
        # Draw all labels on a copy of the original image
        processed_image = original_image.copy()
        processed_image = draw_detection_labels(processed_image, results['detections'])
        results['processed_image'] = processed_image
        
        # Determine overall severity (highest stage detected)
        results['detections'].sort(key=lambda x: (x['box'][0], x['box'][1])) # Sort spatially for consistency
        max_stage_idx = max([det['stage_idx'] for det in results['detections']])
        highest_stage_detection = next(det for det in results['detections'] if det['stage_idx'] == max_stage_idx)
        results['overall_severity'] = highest_stage_detection['stage_name']
    else:
        results['processed_image'] = original_image # No changes
        
    return results


def visualize_results(image_path, results, output_path=None):
    """
    Visualize detection and staging results on the image
    Args:
        image_path: Path to input image
        results: Results dictionary from predict_with_staging
        output_path: Path to save the output image (optional)
    Returns:
        PIL.Image: Image with visualized results
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw detection boxes with stage information
    for detection in results['detections']:
        x1, y1, x2, y2 = detection['box']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Convert hex color to BGR
        hex_color = detection['stage_color']
        bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))[::-1]  # Convert hex to BGR
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, 2)
        
        # Prepare label text
        label = f"{detection['stage_name']} ({detection['stage_confidence']:.2f})"
        confidence_text = f"Conf: {detection['confidence']:.2f}"
        
        # Get text size to draw background rectangle
        (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (w2, h2), _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        max_w = max(w1, w2)
        total_h = h1 + h2 + 10
        
        # Draw background rectangle for text
        cv2.rectangle(image, (x1, y1 - total_h), (x1 + max_w + 10, y1), bgr_color, -1)
        
        # Put text on image
        cv2.putText(image, label, (x1 + 5, y1 - h2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, confidence_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Convert back to PIL Image
    result_image = Image.fromarray(image)
    
    # Save if output path is provided
    if output_path:
        result_image.save(output_path)
    
    return result_image


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_with_staging.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Run two-stage prediction
    results = predict_with_staging(image_path)
    
    # Print results
    print(f"Results for {image_path}:")
    print(f"Overall Severity: {results['overall_severity']}")
    print(f"Number of ulcers detected: {len(results['detections'])}")
    
    for i, detection in enumerate(results['detections']):
        print(f"  Ulcer {i+1}:")
        print(f"    BBox: {detection['box']}")
        print(f"    Detection Confidence: {detection['confidence']:.3f}")
        print(f"    Stage: {detection['stage_name']}")
        print(f"    Stage Description: {detection['stage_description']}")
        print(f"    Stage Confidence: {detection['stage_confidence']:.3f}")
    
    # Visualize results
    output_path = image_path.replace('.', '_staged.')
    visualize_results(image_path, results, output_path)
    print(f"Visualization saved to: {output_path}")
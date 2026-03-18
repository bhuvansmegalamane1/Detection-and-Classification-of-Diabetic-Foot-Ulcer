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
    def __init__(self, config_path="stage_config.yaml"):
        """
        Initialize the ulcer staging classifier
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load the trained stage classification model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        Load the trained stage classification model
        """
        # Initialize the model architecture
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 
                                      self.config['stage_classifier']['num_classes'])
        
        # Load the trained weights
        checkpoint = torch.load(self.config['stage_classifier']['model_path'], 
                               map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model

    def predict_stage(self, cropped_region):
        """
        Predict the stage of a cropped ulcer region
        Args:
            cropped_region: PIL Image of the cropped ulcer region
        Returns:
            tuple: (predicted_stage, confidence_score)
        """
        # Preprocess the image
        input_tensor = self.transform(cropped_region).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            stage_idx = predicted.item()
            stage_confidence = confidence.item()
            
            # Get stage name and description
            stage_name = self.config['stage_names'][str(stage_idx)]
            stage_description = self.config['stage_descriptions'][str(stage_idx)]
            stage_color = self.config['stage_colors'][str(stage_idx)]
        
        return stage_idx, stage_name, stage_description, stage_color, stage_confidence


def predict_with_staging(image_path, detection_model_path="runs/detect/yolov8m_custom/weights/best.pt", 
                        config_path="stage_config.yaml"):
    """
    Run two-stage inference: detection followed by staging
    Args:
        image_path: Path to input image
        detection_model_path: Path to YOLOv8 detection model
        config_path: Path to configuration file
    Returns:
        dict: Combined detection and staging results
    """
    # Initialize the detection model
    detection_model = YOLO(detection_model_path)
    
    # Initialize the stage classifier
    stage_classifier = UlcerStageClassifier(config_path)
    
    # Load the input image
    original_image = Image.open(image_path).convert('RGB')
    
    # Stage 1: Run ulcer detection
    detection_results = detection_model.predict(
        source=image_path,
        conf=0.2,  # Lowered confidence threshold for better detection
        iou=0.45,  # IoU threshold for NMS
        imgsz=640,  # Image size
        augment=True,  # Augmented inference for better accuracy
        agnostic_nms=False,  # Class-agnostic NMS
        max_det=300,  # Maximum detections per image
        save_conf=True  # Save confidence scores
    )
    
    detection_result = detection_results[0]
    boxes = detection_result.boxes
    
    results = {
        'image_path': image_path,
        'detections': [],
        'overall_severity': 'None'
    }
    
    if boxes is not None and len(boxes) > 0:
        # Get all predictions and confidences
        confidences = np.array(boxes.conf.cpu())
        classes = np.array(boxes.cls.cpu())
        bboxes = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
        
        # Process each detected ulcer
        for i in range(len(boxes)):
            x1, y1, x2, y2 = bboxes[i]
            detection_conf = confidences[i]
            
            # Crop the ulcer region from the original image
            cropped_region = original_image.crop((x1, y1, x2, y2))
            
            # Stage 2: Classify the stage of the ulcer
            stage_idx, stage_name, stage_description, stage_color, stage_conf = \
                stage_classifier.predict_stage(cropped_region)
            
            # Store detection and staging results
            detection_info = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'detection_confidence': float(detection_conf),
                'stage_idx': int(stage_idx),
                'stage_name': stage_name,
                'stage_description': stage_description,
                'stage_color': stage_color,
                'stage_confidence': float(stage_conf)
            }
            
            results['detections'].append(detection_info)
        
        # Determine overall severity (highest stage detected)
        max_stage_idx = max([det['stage_idx'] for det in results['detections']])
        # Find the detection with the highest stage
        highest_stage_detection = next(det for det in results['detections'] if det['stage_idx'] == max_stage_idx)
        results['overall_severity'] = highest_stage_detection['stage_name']
        
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
        x1, y1, x2, y2 = detection['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Convert hex color to BGR
        hex_color = detection['stage_color']
        bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))[::-1]  # Convert hex to BGR
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, 2)
        
        # Prepare label text
        label = f"{detection['stage_name']} ({detection['stage_confidence']:.2f})"
        confidence_text = f"Conf: {detection['detection_confidence']:.2f}"
        
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
        print(f"    BBox: {detection['bbox']}")
        print(f"    Detection Confidence: {detection['detection_confidence']:.3f}")
        print(f"    Stage: {detection['stage_name']}")
        print(f"    Stage Description: {detection['stage_description']}")
        print(f"    Stage Confidence: {detection['stage_confidence']:.3f}")
    
    # Visualize results
    output_path = image_path.replace('.', '_staged.')
    visualize_results(image_path, results, output_path)
    print(f"Visualization saved to: {output_path}")
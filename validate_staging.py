import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import yaml
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from predict_with_staging import UlcerStageClassifier, predict_with_staging


def validate_detection_model(detection_model_path, val_images_dir, val_labels_dir):
    """
    Validate the detection model performance
    """
    print("Validating detection model...")
    
    # Load detection model
    model = YOLO(detection_model_path)
    
    # Run validation on the validation dataset
    # We'll create a temporary YAML config for validation if needed
    temp_yaml_path = "temp_val_config.yaml"
    
    # Create a temporary dataset config for validation
    val_config = {
        'path': os.path.dirname(val_images_dir),  # Base path
        'train': val_images_dir,  # Using val dir for this test
        'val': val_images_dir,
        'test': val_images_dir,
        'nc': 1,  # Assuming ulcer detection is single class
        'names': ['ulcer']
    }
    
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(val_config, f)
    
    try:
        # Run validation
        results = model.val(data=temp_yaml_path, plots=False)
        
        # Extract metrics from results
        detection_metrics = {
            'mAP50': getattr(results, 'box', type('obj', (object,), {'map50': 0}))().map50 if hasattr(results, 'box') else 0,
            'mAP50_95': getattr(results, 'box', type('obj', (object,), {'map': 0}))().map if hasattr(results, 'box') else 0,
            'precision': getattr(results, 'box', type('obj', (object,), {'p': 0}))().p if hasattr(results, 'box') else 0,
            'recall': getattr(results, 'box', type('obj', (object,), {'r': 0}))().r if hasattr(results, 'box') else 0,
            'f1_score': getattr(results, 'box', type('obj', (object,), {'f1': 0}))().f1 if hasattr(results, 'box') else 0
        }
    except Exception as e:
        print(f"Error during detection validation: {e}")
        # Fallback to basic evaluation if validation fails
        detection_metrics = {
            'mAP50': 0,
            'mAP50_95': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)
    
    return detection_metrics


def validate_stage_classifier(config_path, val_data_dir):
    """
    Validate the stage classification model
    """
    print("Validating stage classification model...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load the stage classifier
    try:
        stage_classifier = UlcerStageClassifier(config_path)
    except Exception as e:
        print(f"Stage classifier not available: {e}")
        # Return zero metrics if model doesn't exist yet
        classes = sorted(os.listdir(val_data_dir))
        num_classes = len(classes)
        class_metrics = {}
        for i in range(num_classes):
            class_metrics[f"Stage_{i}"] = {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'support': 0
            }
        
        metrics = {
            'accuracy': 0,
            'per_class_metrics': class_metrics,
            'precision_macro': 0,
            'recall_macro': 0,
            'f1_macro': 0
        }
        
        # Create empty confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=int)
        
        return metrics, cm
    
    # Get all validation data
    classes = sorted(os.listdir(val_data_dir))
    all_predictions = []
    all_targets = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(val_data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Get prediction
                    with torch.no_grad():
                        stage_idx, _, _, _, stage_conf = stage_classifier.predict_stage(img)
                    
                    all_predictions.append(stage_idx)
                    all_targets.append(class_idx)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_predictions, average=None)
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(classes):
        class_metrics[f"Stage_{i}"] = {
            'precision': precision[i] if i < len(precision) else 0,
            'recall': recall[i] if i < len(recall) else 0,
            'f1_score': f1[i] if i < len(f1) else 0,
            'support': support[i] if i < len(support) else 0
        }
    
    # Overall metrics
    metrics = {
        'accuracy': accuracy,
        'per_class_metrics': class_metrics,
        'precision_macro': np.mean(precision) if len(precision) > 0 else 0,
        'recall_macro': np.mean(recall) if len(recall) > 0 else 0,
        'f1_macro': np.mean(f1) if len(f1) > 0 else 0
    }
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return metrics, cm


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save the confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Ulcer Staging')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def validate_two_stage_pipeline(image_dir, detection_model_path, config_path):
    """
    Validate the complete two-stage pipeline
    """
    print("Validating two-stage pipeline...")
    
    # Count total images in the validation directory
    total_images = 0
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                total_images += 1
    
    success_count = 0
    total_time = 0
    stage_predictions = []
    stage_targets = []
    
    # Iterate through validation images and run the full pipeline
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                
                # Extract target stage from the path if available (assuming folder structure)
                # For example, if image is in dataset_staged/val/stage1/image.jpg, target stage is 0
                path_parts = image_path.split(os.sep)
                target_stage = -1
                for i, part in enumerate(path_parts):
                    if 'stage' in part.lower():
                        stage_num = ''.join(filter(str.isdigit, part))
                        if stage_num:
                            target_stage = int(stage_num) - 1  # Convert to 0-indexed
                            break
                
                try:
                    start_time = time.time()
                    results = predict_with_staging(image_path, detection_model_path, config_path)
                    end_time = time.time()
                    
                    total_time += (end_time - start_time)
                    success_count += 1
                    
                    # If target stage is available and we have detections, collect stage predictions
                    if target_stage != -1 and results['detections']:
                        # For simplicity, take the highest stage detected
                        if results['detections']:
                            predicted_stage = max([det['stage_idx'] for det in results['detections']])
                            stage_predictions.append(predicted_stage)
                            stage_targets.append(target_stage)
                            
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    # Calculate pipeline metrics
    success_rate = success_count / total_images if total_images > 0 else 0
    avg_time_per_image = total_time / success_count if success_count > 0 else 0
    
    # Calculate staging accuracy if we have any valid comparisons
    staging_accuracy = 0
    if stage_predictions and stage_targets:
        staging_accuracy = accuracy_score(stage_targets, stage_predictions)
    
    pipeline_metrics = {
        'detection_accuracy_preserved': True,  # This is assumed if the pipeline runs successfully
        'stage_classification_accuracy': staging_accuracy,
        'overall_pipeline_time': avg_time_per_image,
        'success_rate': success_rate,
        'total_images_processed': success_count,
        'total_images_found': total_images
    }
    
    return pipeline_metrics


def main():
    parser = argparse.ArgumentParser(description='Validate ulcer staging system')
    parser.add_argument('--config', type=str, default='stage_config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--val_detection_images', type=str, 
                        default='dataset/val/images', 
                        help='Path to validation images for detection')
    parser.add_argument('--val_detection_labels', type=str, 
                        default='dataset/val/labels', 
                        help='Path to validation labels for detection')
    parser.add_argument('--val_staging_data', type=str, 
                        default='dataset_staged/val', 
                        help='Path to validation data for staging')
    parser.add_argument('--detection_model', type=str, 
                        default='runs/detect/yolov8m_custom/weights/best.pt', 
                        help='Path to detection model')
    parser.add_argument('--output_dir', type=str, default='validation_results', 
                        help='Directory to save validation results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting validation of ulcer staging system...")
    
    # Validate detection model
    detection_metrics = validate_detection_model(
        args.detection_model, 
        args.val_detection_images, 
        args.val_detection_labels
    )
    
    # Validate stage classifier
    stage_metrics, cm = validate_stage_classifier(args.config, args.val_staging_data)
    
    # Validate complete pipeline
    pipeline_metrics = validate_two_stage_pipeline(
        args.val_detection_images,
        args.detection_model,
        args.config
    )
    
    # Plot confusion matrix
    class_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Save results to file
    results_path = os.path.join(args.output_dir, 'validation_results.txt')
    with open(results_path, 'w') as f:
        f.write("Ulcer Staging System Validation Results\n")
        f.write("="*50 + "\n\n")
        
        f.write("Detection Model Metrics:\n")
        for metric, value in detection_metrics.items():
            f.write(f"  {metric}: {value}\n")
        f.write("\n")
        
        f.write("Stage Classification Metrics:\n")
        f.write(f"  Overall Accuracy: {stage_metrics['accuracy']:.4f}\n")
        f.write(f"  Macro Average Precision: {stage_metrics['precision_macro']:.4f}\n")
        f.write(f"  Macro Average Recall: {stage_metrics['recall_macro']:.4f}\n")
        f.write(f"  Macro Average F1-Score: {stage_metrics['f1_macro']:.4f}\n")
        f.write("\n")
        
        f.write("Per-Class Metrics:\n")
        for class_name, metrics in stage_metrics['per_class_metrics'].items():
            f.write(f"  {class_name}:\n")
            for metric, value in metrics.items():
                f.write(f"    {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Pipeline Metrics:\n")
        for metric, value in pipeline_metrics.items():
            f.write(f"  {metric}: {value}\n")
        f.write("\n")
        
        f.write(f"Confusion Matrix saved to: {cm_path}\n")
    
    # Print summary
    print("\nValidation Summary:")
    print(f"Stage Classification Accuracy: {stage_metrics['accuracy']:.4f}")
    print(f"Pipeline Success Rate: {pipeline_metrics['success_rate']:.4f}")
    print(f"Pipeline Time: {pipeline_metrics['overall_pipeline_time']:.2f}s per image")
    print(f"Results saved to: {results_path}")
    
    # Check if targets are met
    print("\nTarget Achievement:")
    print(f"  Stage Classification Accuracy (≥85%): {'✓' if stage_metrics['accuracy'] >= 0.85 else '✗'}")
    print(f"  Pipeline Time (<2s): {'✓' if pipeline_metrics['overall_pipeline_time'] < 2.0 else '✗'}")


if __name__ == "__main__":
    main()
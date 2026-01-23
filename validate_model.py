from ultralytics import YOLO
import os

def validate_model():
    """
    Validate the trained model to check its accuracy
    """
    try:
        # Try to load the best trained model
        model_path = "runs/detect/yolov8m_custom_very_high_accuracy/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Trying alternative model paths...")
            
            # Try other possible model paths
            alternative_paths = [
                "runs/detect/yolov8m_custom_high_accuracy/weights/best.pt",
                "runs/detect/yolov8m_custom/weights/best.pt",
                "best.pt"
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Found model at {model_path}")
                    break
            else:
                print("No trained model found. Please train the model first.")
                return

        # Load the model
        model = YOLO(model_path)
        
        # Validate the model on the validation set
        print("Validating model on validation dataset...")
        metrics = model.val(
            data='datacustom.yaml',
            imgsz=640,
            batch=16,
            conf=0.001,  # Low confidence threshold for comprehensive evaluation
            iou=0.6,  # IoU threshold for NMS
            max_det=300,  # Maximum detections per image
            half=False,  # Use full precision
            device='',  # Use GPU if available
            augment=True  # Augmented inference
        )
        
        # Print validation results
        print("\n" + "="*50)
        print("MODEL VALIDATION RESULTS")
        print("="*50)
        print(f"mAP50: {metrics.box.map50:.4f}")  # mAP50
        print(f"mAP50-95: {metrics.box.map:.4f}")  # mAP50-95
        print(f"Precision: {metrics.box.p:.4f}")  # Precision
        print(f"Recall: {metrics.box.r:.4f}")  # Recall
        print(f"F1-Score: {metrics.box.f1:.4f}")  # F1 Score
        print("="*50)
        
        # Save results
        metrics.save_dir = "validation_results"
        print(f"Validation results saved to: {metrics.save_dir}")
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_model()
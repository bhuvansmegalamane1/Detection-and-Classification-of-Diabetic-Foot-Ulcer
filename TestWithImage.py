from ultralytics import YOLO
import os
import numpy as np

def run_inference():
    """
    Run inference on a single image with improved accuracy settings
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
        
        # Check if test image exists
        test_image = "3.jpg"
        if not os.path.exists(test_image):
            print(f"Test image not found at {test_image}")
            # List available images
            images = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image = images[0]
                print(f"Using {test_image} instead")
            else:
                print("No images found in current directory")
                return

        # Perform prediction with enhanced parameters for better accuracy
        print(f"Running inference on {test_image}...")
        results = model.predict(
            source=test_image,
            conf=0.25,  # Confidence threshold
            iou=0.45,  # IoU threshold for NMS
            imgsz=640,  # Image size
            show=True,  # Show results
            save=True,  # Save results
            save_txt=True,  # Save labels
            save_conf=True,  # Save confidence scores
            save_crop=False,  # Don't save cropped images
            show_labels=True,  # Show labels
            show_conf=True,  # Show confidence scores
            max_det=300,  # Maximum detections per image
            augment=True,  # Augmented inference
            agnostic_nms=False,  # Class-agnostic NMS
            verbose=True  # Verbose output
        )

        # Process results
        result = results[0]
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            print(f"\nFound {len(boxes)} detection(s):")
            # Extract confidence and class information
            # Convert to numpy arrays directly
            confidences = np.array(boxes.conf)
            classes = np.array(boxes.cls)
            
            for i in range(len(boxes)):
                confidence = float(confidences[i])
                cls = int(classes[i])
                class_name = model.names[cls]
                print(f"  Detection {i+1}: {class_name} with {confidence:.3f} confidence")
        else:
            print("No detections found")
            
        print(f"Results saved to: {result.save_dir}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_inference()
from ultralytics import YOLO
import os

# Ensure the datacustom.yaml file is properly configured
# Check if the dataset paths exist
if not os.path.exists(r'e:\foot\Foot_Ulcer\Foot_Ulcer\dataset\train\images'):
    print("Warning: Training dataset path does not exist!")
if not os.path.exists(r'e:\foot\Foot_Ulcer\Foot_Ulcer\dataset\val\images'):
    print("Warning: Validation dataset path does not exist!")

# Load the YOLOv8 model (using a larger model for higher accuracy)
# You can try yolov8l.pt or yolov8x.pt for even higher accuracy (but slower training/inference)
model = YOLO('yolov8m.pt')  # Using pretrained weights for transfer learning

print("Starting high accuracy training...")

# Train the model with improved parameters for higher accuracy
results = model.train(
    data='datacustom.yaml',  # Updated path
    epochs=50,  # Increased epochs for better convergence
    imgsz=640,  # Standard image size
    batch=16,  # Batch size
    name='yolov8m_custom_very_high_accuracy',  # New experiment name
    patience=15,  # Early stopping patience
    optimizer='AdamW',  # Advanced optimizer
    lr0=0.001,  # Initial learning rate
    lrf=0.0001,  # Final learning rate
    momentum=0.937,  # Momentum for SGD
    weight_decay=0.0005,  # Weight decay
    warmup_epochs=5,  # Warmup epochs
    warmup_momentum=0.8,  # Warmup momentum
    box=7.5,  # Box loss gain
    cls=0.5,  # Class loss gain
    dfl=1.5,  # Distribution focal loss gain
    hsv_h=0.015,  # Image HSV-Hue augmentation
    hsv_s=0.7,  # Image HSV-Saturation augmentation
    hsv_v=0.4,  # Image HSV-Value augmentation
    degrees=0.0,  # Image rotation (+/- deg)
    translate=0.1,  # Image translation (+/- fraction)
    scale=0.5,  # Image scale (+/- gain)
    shear=0.0,  # Image shear (+/- deg)
    perspective=0.0,  # Image perspective (+/- fraction)
    flipud=0.0,  # Image flip up-down (probability)
    fliplr=0.5,  # Image flip left-right (probability)
    mosaic=1.0,  # Image mosaic (probability)
    mixup=0.1,  # Image mixup (probability) - helps with generalization
    copy_paste=0.1,  # Segment copy-paste (probability) - helps with generalization
    seed=0,  # Random seed
    deterministic=False,  # Deterministic training
    verbose=True,  # Verbose output
    pretrained=True,  # Use pretrained weights
    augment=True,  # Augment training data
    rect=False,  # Rectangular training
    cache=True,  # Cache images for faster training
    single_cls=False,  # Treat as single-class dataset
    overlap_mask=True,  # Mask overlap
    mask_ratio=4,  # Mask downsample ratio
    dropout=0.0,  # Dropout probability
    val=True,  # Validate during training
    save_period=10,  # Save checkpoint every N epochs
)

print("Training completed! Model saved in runs/detect/yolov8m_custom_very_high_accuracy/")
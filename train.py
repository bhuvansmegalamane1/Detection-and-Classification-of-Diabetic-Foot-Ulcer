from ultralytics import YOLO

# Load the YOLOv8m model (you can also try YOLOv8l or YOLOv8x for higher accuracy)
model = YOLO('yolov8m.pt')  # Using pretrained weights for transfer learning

# Train the model with improved parameters for higher accuracy
results = model.train(
    data='datacustom.yaml',  # Updated path
    epochs=50,  # Increased epochs for better convergence
    imgsz=640,  # Standard image size
    batch=16,  # Batch size
    name='yolov8m_custom_high_accuracy',  # New experiment name
    patience=10,  # Early stopping patience
    optimizer='AdamW',  # Advanced optimizer
    lr0=0.001,  # Initial learning rate
    lrf=0.0001,  # Final learning rate
    momentum=0.937,  # Momentum for SGD
    weight_decay=0.0005,  # Weight decay
    warmup_epochs=3,  # Warmup epochs
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
    mixup=0.0,  # Image mixup (probability)
    copy_paste=0.0,  # Segment copy-paste (probability)
    seed=0,  # Random seed
    deterministic=False,  # Deterministic training
    verbose=True  # Verbose output
)
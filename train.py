from ultralytics import YOLO

# Load the YOLOv8m model
model = YOLO('yolov8m.pt')  # Use 'yolov8m.yaml' to build from scratch

# Train the model
results = model.train(
    data='D:\YoloAllProjects\Projects\Foot_Ulcer\datacustom.yaml',
    epochs=5,
    imgsz=640,
    batch=16,
    name='yolov8m_custom'
)

from ultralytics import YOLO
model=YOLO("best.pt")
model.predict(mode="predict", model="best.pt", show=True,conf=0.5,save=True, source="3.jpg")

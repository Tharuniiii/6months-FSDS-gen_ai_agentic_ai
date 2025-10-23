from ultralytics import YOLO

# ✅ This automatically downloads YOLOv11n.pt
model = YOLO("yolov11n.pt")

# Predict
results = model.predict(
    source="Users\Tharuni\Desktop\NIT\oct month\21st_ultralytics'\download (1).jpeg",
    conf=0.25,
    save=True
)

print(results)

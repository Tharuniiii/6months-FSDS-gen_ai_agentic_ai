import cv2
from ultralytics import YOLO

# ✅ Load your trained YOLOv8 segmentation model
model = YOLO(r"C:\Users\Tharuni\Desktop\NIT\oct month\21st,22nd_ultralytics'\stationary\runs\segment\train\weights\best.pt")

# ✅ Open webcam (use 0 for default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ YOLOv8 Segmentation model loaded successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error: Could not read frame.")
        break

    # Run segmentation on the frame
    results = model.predict(frame)

    # Draw segmentation masks and bounding boxes
    annotated_frame = results[0].plot()

    # Display result
    cv2.imshow("🟢 YOLOv8 Segmentation - Stationary Items", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

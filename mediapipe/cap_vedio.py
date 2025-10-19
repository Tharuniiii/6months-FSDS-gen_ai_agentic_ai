print("✅ Script started running")

import cv2
import mediapipe as mp
import os

# ----> Initialize MediaPipe modules
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# ----> Path to your video file
video_path = r"C:\Users\Tharuni\Desktop\NIT\oct month\15th_mediapipe\5385949-uhd_2160_4096_30fps.mp4"

# ----> Check if file exists
if not os.path.exists(video_path):
    print("⚠️ Video file not found! Please check the path.")
else:
    print("🎥 Video found, starting detection...")

# ----> Capture the video
cap = cv2.VideoCapture(video_path)
print("Video opened:", cap.isOpened())

# ----> Create Objectron model
objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=10,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.7,
    model_name='Chair'
)

# ----> Create a visible window
cv2.namedWindow("MediaPipe Objectron", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MediaPipe Objectron", 800, 600)

# ----> Process video frames
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⛔ End of video or cannot read frame.")
        break

    # Convert color space
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw results
    if results.detected_objects:
        print(f"✅ Detected {len(results.detected_objects)} object(s).")
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
                image,
                detected_object.landmarks_2d,
                mp_objectron.BOX_CONNECTIONS
            )
            mp_drawing.draw_axis(
                image,
                detected_object.rotation,
                detected_object.translation
            )

    # Show the image
    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))

    # Quit with 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Program finished.")

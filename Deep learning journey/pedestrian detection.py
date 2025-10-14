import cv2
import numpy as np
import os

# Set the path for the body classifier (Haar Cascade XML)
body_classifier_path = r'C:\Users\Admin\AVSCODE\7. OPENCV\haar cascade classifier basic project\Haarcascades\haarcascade_fullbody.xml'

# Check if the classifier path exists
if not os.path.exists(body_classifier_path):
    print(f"Error: The classifier file does not exist at {body_classifier_path}")
    exit()

# Create the body classifier using the provided path
body_classifier = cv2.CascadeClassifier(body_classifier_path)

# Check if the classifier is loaded successfully
if body_classifier.empty():
    print("Error: Could not load the body classifier. Make sure the XML file is valid and accessible.")
    exit()

# Set the path for the video file
video_path = r'C:\Users\Admin\Desktop\MyFile\0. DATASCIENCE PROJECT\10. Computer vision\Computer-Vision-Tutorial-master\Computer-Vision-Tutorial-master\image_examples\walking.avi'

# Check if the video path exists
if not os.path.exists(video_path):
    print(f"Error: The video file does not exist at {video_path}")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

print("Video opened successfully. Starting pedestrian detection...")

while cap.isOpened():
    ret, frame = cap.read()

    # If frame is not read correctly, exit the loop
    if not ret:
        print("Error: Failed to read frame from video. Exiting...")
        break

    # Convert the frame to grayscale for better detection performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the grayscale image
    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw bounding boxes for detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the result with detected bodies
    cv2.imshow('Pedestrians', frame)

    # Press Enter (13) to exit
    if cv2.waitKey(1) == 13:  # 13 corresponds to the Enter key
        print("Exiting...")
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

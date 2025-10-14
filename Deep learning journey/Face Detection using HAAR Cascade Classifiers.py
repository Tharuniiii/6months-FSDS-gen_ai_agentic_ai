import numpy as np
import cv2

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(r"C:\Users\Admin\AVSCODE\7. OPENCV\haar cascade classifier basic project\Haarcascades\haarcascade_frontalface_default.xml")
# Load the image
image = cv2.imread(r"C:\Users\Admin\Desktop\NIT\1. NIT_Batches\10. WORKSHOP\2024 - WORKSHOP\7. Exploring Generative AI through computer vision\myimage.jpg")

#image = cv2.imread(r'C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\2. DATASCIENCE PROJECT\10. Computer vision\Computer-Vision-Tutorial-master\Computer-Vision-Tutorial-master\image_examples\5.jpg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()  # Exit if image is not loaded
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Check if faces are detected
if len(faces) == 0:
    print("No faces found!")
else:
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:  # (x, y) is the top-left corner, and (w, h) is the width and height of the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

    # Display the output image
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)  # Wait for a key press to close the window

# Close all OpenCV windows
cv2.destroyAllWindows()

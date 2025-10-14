import cv2

class FaceAndEyeDetection:
    def __init__(self, face_cascade_path, eye_cascade_path):
        # Load the cascade classifiers for face and eyes
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Check if classifiers were loaded properly
        if self.face_cascade.empty():
            print("Error: Could not load face cascade.")
            exit()
        if self.eye_cascade.empty():
            print("Error: Could not load eye cascade.")
            exit()

    def detect_faces(self, gray, frame):
        """Detect faces in the given frame."""
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Get region of interest (ROI) for eye detection
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            self.detect_eyes(roi_gray, roi_color)
        return frame

    def detect_eyes(self, roi_gray, roi_color):
        """Detect eyes in the region of interest (ROI)."""
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    def start_detection(self):
        """Start webcam capture and detect faces and eyes in real-time."""
        # Capture video from webcam
        video_capture = cv2.VideoCapture(0)

        # Check if webcam is opened correctly
        if not video_capture.isOpened():
            print("Error: Could not access the webcam.")
            exit()

        while True:
            # Read frame-by-frame from the video capture
            ret, frame = video_capture.read()

            # If frame is read correctly, process it
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces and eyes in the current frame
            canvas = self.detect_faces(gray, frame)

            # Display the resulting frame
            cv2.imshow('Face and Eye Detection', canvas)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

        # Release the video capture and close all OpenCV windows
        video_capture.release()
        cv2.destroyAllWindows()

# Paths to Haar cascade XML files for face and eye detection
face_cascade_path = r'C:\Users\Admin\AVSCODE\7. OPENCV\haar cascade classifier basic project\Haarcascades\haarcascade_frontalface_default.xml'
eye_cascade_path = r'C:\Users\Admin\AVSCODE\7. OPENCV\haar cascade classifier basic project\Haarcascades\haarcascade_eye.xml'

# Initialize the Face and Eye Detection system
detection_system = FaceAndEyeDetection(face_cascade_path, eye_cascade_path)

# Start face and eye detection
detection_system.start_detection()

import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create a blank canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Track finger state
drawing = False
prev_x, prev_y = 0, 0

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmarks for index fingertip and thumb tip
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            ix, iy = int(index_finger.x * w), int(index_finger.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Distance between thumb and index finger
            distance = euclidean_distance((ix, iy), (tx, ty))

            # Gesture math: if distance is small, pinch (draw mode)
            if distance < 40:
                drawing = True
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), (255, 0, 0), 5)
                prev_x, prev_y = ix, iy
            else:
                drawing = False
                prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay canvas on frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Instructions
    cv2.putText(frame, 'Pinch (Thumb + Index) to Draw', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow("Virtual Painter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

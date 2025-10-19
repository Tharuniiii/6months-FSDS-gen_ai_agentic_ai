import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Variables
draw_color = (255, 0, 255)  # Default: Purple
brush_thickness = 7
eraser_thickness = 50
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)

# Open Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Finger tips
tip_ids = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks):
    fingers = []
    # Thumb
    fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    # Fingers
    for id in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y else 0)
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if lm_list:
                x1, y1 = lm_list[8]  # Index tip
                x2, y2 = lm_list[12]  # Middle tip

                fingers = fingers_up(handLms)

                # Selection Mode: Two fingers up
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    cv2.rectangle(img, (x1, y1-25), (x2, y2+25), draw_color, cv2.FILLED)
                    # Change color based on x1 position
                    if y1 < 100:
                        if 250 < x1 < 450:
                            draw_color = (255, 0, 255)  # Purple
                        elif 550 < x1 < 750:
                            draw_color = (0, 255, 0)    # Green
                        elif 800 < x1 < 950:
                            draw_color = (0, 0, 255)    # Red
                        elif 1000 < x1 < 1200:
                            draw_color = (0, 0, 0)      # Eraser

                # Drawing Mode: Index finger up
                elif fingers[1] and not fingers[2]:
                    cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    if draw_color == (0, 0, 0):
                        cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                        cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                        cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                    xp, yp = x1, y1

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Merge canvas and webcam feed
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # Toolbar
    cv2.rectangle(img, (250, 0), (450, 100), (255, 0, 255), cv2.FILLED)
    cv2.rectangle(img, (550, 0), (750, 100), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (800, 0), (950, 100), (0, 0, 255), cv2.FILLED)
    cv2.rectangle(img, (1000, 0), (1200, 100), (0, 0, 0), cv2.FILLED)

    cv2.putText(img, "Save: Press 's'", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Air Writing", img)

    key = cv2.waitKey(1)
    if key == ord('s'):
        timestamp = int(time.time())
        cv2.imwrite(f"air_drawing_{timestamp}.png", img_canvas)
        print("Artwork saved!")
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

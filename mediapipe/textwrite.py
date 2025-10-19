import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam Capture
cap = cv2.VideoCapture(0)
canvas = None

# Drawing settings
draw_color = (255, 0, 255)  # Purple
brush_thickness = 8

# Previous point
xp, yp = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror the frame
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if lm_list:
                x1, y1 = lm_list[8]  # Index finger tip
                x2, y2 = lm_list[12] # Middle finger tip

                # Check if only index finger is up
                fingers = []
                for tip_id in [8, 12, 16, 20]:
                    fingers.append(lm_list[tip_id][1] < lm_list[tip_id - 2][1])

                if fingers[0] and not fingers[1]:  # Drawing mode
                    cv2.circle(frame, (x1, y1), 8, draw_color, -1)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0

                # Clear canvas if all fingers up
                if all(fingers):
                    canvas = np.zeros_like(frame)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Merge canvas and frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    frame_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    final = cv2.add(frame_bg, frame_fg)

    cv2.imshow("Virtual Painter", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

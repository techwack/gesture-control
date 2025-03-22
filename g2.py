import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Variables for drawing
drawing = False  # True when drawing is active
current_color = (0, 255, 0)  # Default color (green)
canvas = None  # Canvas to store drawings
prev_x, prev_y = None, None  # Previous coordinates for drawing

# Smoothing variables
smoothing_factor = 0.5  # Higher values make the drawing smoother
smoothed_x, smoothed_y = None, None

# List of colors to choose from
colors = [
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (0, 0, 255),  # Red
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
]

def count_fingers(hand_landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other 4 fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Initialize canvas for drawing
    if canvas is None:
        canvas = np.zeros_like(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            label = hand_info.classification[0].label  # 'Left' or 'Right'
            fingers = count_fingers(hand_landmarks, label)

            # Get landmark positions
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # Smooth the coordinates
            if smoothed_x is None or smoothed_y is None:
                smoothed_x, smoothed_y = x, y
            else:
                smoothed_x = int(smoothing_factor * x + (1 - smoothing_factor) * smoothed_x)
                smoothed_y = int(smoothing_factor * y + (1 - smoothing_factor) * smoothed_y)

            # Freehand drawing logic
            if fingers == 1:  # Index finger up
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (smoothed_x, smoothed_y), current_color, 5)
                prev_x, prev_y = smoothed_x, smoothed_y
            else:
                prev_x, prev_y = None, None

            # Color selection logic
            if fingers == 2:  # Two fingers up
                current_color = colors[0]  # Green
            elif fingers == 3:  # Three fingers up
                current_color = colors[1]  # Blue
            elif fingers == 4:  # Four fingers up
                current_color = colors[2]  # Red
            elif fingers == 5:  # Five fingers up
                current_color = colors[3]  # Yellow

            # Fill shape logic
            if fingers == 0:  # Fist (closed hand)
                # Detect contours on the canvas
                gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Fill the largest contour with the current color
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(canvas, [largest_contour], -1, current_color, -1)

            # Display finger count
            cv2.putText(frame, f'Fingers: {fingers}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, (255, 0, 0), 3)

    # Overlay canvas on the frame
    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)

    # Display the frame
    cv2.imshow("Gesture Control", frame)

    # Key bindings
    key = cv2.waitKey(1)
    if key == ord('e'):  # Erase canvas
        canvas = np.zeros_like(frame)
    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
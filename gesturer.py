import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

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

    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

zoom = 1.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            label = hand_info.classification[0].label
            fingers = count_fingers(hand_landmarks, label)
            cv2.putText(frame, f'Fingers: {fingers}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

            if fingers == 0:
                zoom += 0.01
            elif fingers == 1:
                zoom = max(1.0, zoom - 0.01)
            elif fingers == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif fingers == 3:
                frame = cv2.GaussianBlur(frame, (15, 15), 0)            
            elif fingers == 4:
                frame = cv2.bitwise_not(frame)                

    center = (int(w/2), int(h/2))
    M = cv2.getRotationMatrix2D(center, 0, zoom)
    frame = cv2.warpAffine(frame, M, (w, h))

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
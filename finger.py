import cv2
import mediapipe as mp

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam open
cap = cv2.VideoCapture(0)

# Fingertip landmark IDs
tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            lm_list = []
            h, w, _ = img.shape

            for lm in hand_landmarks.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Finger states: [thumb, index, middle, ring, little]
            fingers = [0, 0, 0, 0, 0]

            # Thumb (x-axis check)
            if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                fingers[0] = 1

            # Other fingers (y-axis check)
            for i in range(1, 5):
                if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
                    fingers[i] = 1

            finger_count = sum(fingers)

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # 🚨 Middle finger only detection
            if fingers == [0, 0, 1, 0, 0]:
                cv2.putText(
                    img,
                    "Aye! gaali kisko dera hai 😡",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    # Show finger count
    cv2.putText(
        img,
        f"Fingers: {finger_count}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

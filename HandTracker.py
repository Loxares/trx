import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Allow detection of up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and find hands
        results = hands.process(image_rgb)

        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box coordinates
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])

                # Convert normalized coordinates to pixel values
                h, w, _ = image.shape
                x_min = int(x_min * w)
                y_min = int(y_min * h)
                x_max = int(x_max * w)
                y_max = int(y_max * h)

                # Draw bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the image
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

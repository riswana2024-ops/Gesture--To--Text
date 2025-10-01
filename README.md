# Gesture--To--Text Sign Language Translator 
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define gesture meanings
GESTURES = {
"Hello": "Hello",
"Yes": "Yes",
"No": "No",
"Thumbs Up": "Good Job",
"Thumbs Down": "Bad",
"Stop": "Stop",
"Peace": "Peace",
"OK": "OK"
}

# Function to check if a finger is open (tip above PIP joint)
def is_finger_open(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y  # True if the tip is above the PIP joint
# Function to recognize gestures based on finger positions
def recognize_gesture(landmarks):
    thumb_tip, index_tip, middle_tip = landmarks[4], landmarks[8], landmarks[12]
    ring_tip, pinky_tip, wrist = landmarks[16], landmarks[20], landmarks[0]

    # Detect open fingers
    thumb_open = landmarks[4].x > landmarks[3].x  # Right hand: Thumb tip is right of thumb base
    index_open = is_finger_open(landmarks, 8, 6)
    middle_open = is_finger_open(landmarks, 12, 10)
    ring_open = is_finger_open(landmarks, 16, 14)
    pinky_open = is_finger_open(landmarks, 20, 18)

    # Recognize gestures
    if index_open and middle_open and ring_open and pinky_open and thumb_open:
        return "Hello"  # Open palm
    elif thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "Thumbs Up" if thumb_tip.y < wrist.y else "Thumbs Down"
    elif not thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "Stop"  # Fist
    elif index_open and middle_open and not ring_open and not pinky_open:
        return "Peace"  # Victory sign (V)
    elif thumb_tip.x < index_tip.x and np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]) < 0.05:
        return "OK"  # Thumb and index touching
    elif index_open and not middle_open and not ring_open and not pinky_open:
        return "Yes"
    elif not index_open and not middle_open and ring_open and pinky_open:
        return "No"
    return None  # No gesture detected

# Initialize video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks.landmark)
                if gesture and gesture in GESTURES:
                    cv2.putText(frame, GESTURES[gesture], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Real-Time Sign Language Translator', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()

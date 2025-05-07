import cv2
import mediapipe as mp

# Initialize Mediapipe solutions
mp_drawing = mp.solutions.drawing_utils                        # For drawing landmarks
mp_hands = mp.solutions.hands                                  # For hand detection and tracking
mp_face_mesh = mp.solutions.face_mesh                          # For face mesh detection

# OpenCV video capture
cap = cv2.VideoCapture(0)  # 0 means default webcam

# Initialize MediaPipe models
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip image to avoid mirror view & convert BGR to RGB
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image for hands and face
        hand_results = hands.process(image)
        face_results = face_mesh.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

        # Show the result
        cv2.imshow('Face and Hand Landmarks', image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
            break

cap.release()
cv2.destroyAllWindows()

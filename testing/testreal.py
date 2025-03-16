import cv2
import numpy as np
import mediapipe as mp
import os
import mysql.connector
from deepface import DeepFace

# Disable TensorFlow oneDNN optimizations (Boosts speed)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Faster processing
    max_num_faces=1,  
    refine_landmarks=True
)

# Initialize Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# OpenCV Capture
cap = cv2.VideoCapture(0)

def detect_open_eyes(face_landmarks):
    """ Check if both eyes are open using facial landmarks """
    left_eye_top = face_landmarks[159].y
    left_eye_bottom = face_landmarks[145].y
    right_eye_top = face_landmarks[386].y
    right_eye_bottom = face_landmarks[374].y
    
    left_eye_open = abs(left_eye_bottom - left_eye_top) > 0.02
    right_eye_open = abs(right_eye_bottom - right_eye_top) > 0.02
    
    return left_eye_open and right_eye_open

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 480))

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face and landmarks
    face_results = face_mesh.process(img_rgb)
    detection_results = face_detector.process(img_rgb)

    if detection_results.detections:
        for detection in detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark

            if detect_open_eyes(face_landmarks):
                cv2.putText(frame, "✅ Face & Eyes Detected!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to Capture", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Check if user presses SPACE to capture
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE key
                    # Save the captured image
                    face_path = "captured_face.jpg"
                    cv2.imwrite(face_path, frame)
                    print("✅ Face captured successfully!")

                    # Extract face embeddings using DeepFace
                    try:
                        embedding = DeepFace.represent(face_path, model_name="ArcFace", detector_backend="mediapipe")[0]["embedding"]

                        # Convert embedding to string
                        embedding_str = ",".join(map(str, embedding))

                        # Store in MySQL database
                        conn = mysql.connector.connect(host="localhost", user="root", password="", database="attendance_system")
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO users (name, face_embedding, image_path) VALUES (%s, %s, %s)", 
                                       ("Student Name", embedding_str, face_path))
                        conn.commit()
                        conn.close()
                        print("✅ Face registered in database!")

                    except Exception as e:
                        print("❌ Error extracting face embeddings:", str(e))

                    break  # Exit loop after capturing

            else:
                cv2.putText(frame, "⚠ Open Your Eyes!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    else:
        cv2.putText(frame, "No Face Detected!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face Registration", frame)

    # Press 'q' to exit without registering
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

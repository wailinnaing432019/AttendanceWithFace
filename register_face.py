import cv2
import numpy as np
import os
from deepface import DeepFace
from database import connect_db

# Directory to save images
IMAGE_DIR = "images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def register_face(name):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle

        cv2.imshow("Face Registration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            image_path = f"{IMAGE_DIR}/{name}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"✅ Image saved: {image_path}")
            cap.release()
            cv2.destroyAllWindows()
            process_face(name, image_path)
            return
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_face(name, image_path):
    try:
        embedding = DeepFace.represent(image_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
        embedding_str = ",".join(map(str, embedding))  # Convert list to string
        
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, face_embedding, image_path) VALUES (%s, %s, %s)",
                       (name, embedding_str, image_path))
        conn.commit()
        conn.close()

        print(f"✅ Face registered for {name}")
    
    except Exception as e:
        print(f"❌ Error processing face: {e}")

if __name__ == "__main__":
    student_name = input("Enter student name: ")
    register_face(student_name)

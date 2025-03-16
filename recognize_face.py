import cv2
import numpy as np
import os
from deepface import DeepFace
from database import connect_db
import datetime

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load registered face embeddings from MySQL
def load_registered_faces():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, face_embedding FROM users")
    users = cursor.fetchall()
    conn.close()

    registered_faces = []
    for user in users:
        user_id, name, embedding_str = user
        embedding = np.array([float(x) for x in embedding_str.split(",")])  # Convert string to numpy array
        registered_faces.append({"id": user_id, "name": name, "embedding": embedding})

    return registered_faces

# Compare captured face with registered faces
def recognize_face(embedding, registered_faces, threshold=15):
    for user in registered_faces:
        distance = np.linalg.norm(user["embedding"] - embedding)  # Euclidean distance
        if distance < threshold:
            return user  # Return matched user
    return None  # No match found

# Mark attendance in MySQL
def mark_attendance(user_id):
    conn = connect_db()
    cursor = conn.cursor()

    # Check if attendance was already recorded in the last 5 minutes
    cursor.execute("""
        SELECT timestamp FROM attendance 
        WHERE user_id = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, (user_id,))
    
    last_record = cursor.fetchone()
    
    if last_record:
        last_time = last_record[0]
        current_time = datetime.datetime.now()
        time_difference = (current_time - last_time).total_seconds() / 60  # Convert to minutes

        if time_difference < 5:  # Ignore duplicate records within 5 minutes
            print(f"⚠️ Attendance already recorded for User ID: {user_id} recently.")
            return
    
    # Insert new attendance record
    cursor.execute("INSERT INTO attendance (user_id) VALUES (%s)", (user_id,))
    conn.commit()
    conn.close()
    print(f"✅ Attendance marked for User ID: {user_id}")
def start_recognition():
    registered_faces = load_registered_faces()
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]  # Extract face region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle

            # Save temporary face image
            temp_image_path = "temp_face.jpg"
            cv2.imwrite(temp_image_path, face_roi)

            try:
                embedding = DeepFace.represent(temp_image_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
                matched_user = recognize_face(embedding, registered_faces)

                if matched_user:
                    cv2.putText(frame, f"{matched_user['name']}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    mark_attendance(matched_user["id"])
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            except Exception as e:
                print(f"❌ Error in recognition: {e}")

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_recognition()

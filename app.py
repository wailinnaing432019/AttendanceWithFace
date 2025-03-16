import cv2
import os
import numpy as np
from deepface import DeepFace
import mysql.connector
import mediapipe as mp
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for,session
from PIL import Image
import imghdr
app = Flask(__name__)
app.secret_key = 'your_secret_key'
IMAGE_PATH = "static/registered_faces"  # Ensure images are stored here

# camera_source = "http://192.168.100.31:4747/video"
camera_source = "http://192.168.152.51:4747/video"

# mp_face_detection = mp.solutions.face_detection
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="attendance_system"
    )

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)  # ✅ Create an instance
def get_facial_landmarks(image, detection):
    height, width, _ = image.shape
    face_mesh_results = mp_face_mesh.process(image)  # Process the image

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            landmark_coords = []

            # Extract all facial landmark points
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmark_coords.append((x, y))

            return landmark_coords  # ✅ Return full list of facial points

    return None  # No face detected


def calculate_ear(landmarks):
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]  # Left eye keypoints
    right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]  # Right eye keypoints

    # Compute vertical distances
    left_ear = (distance(left_eye[1], left_eye[5]) + distance(left_eye[2], left_eye[4])) / (2 * distance(left_eye[0], left_eye[3]))
    right_ear = (distance(right_eye[1], right_eye[5]) + distance(right_eye[2], right_eye[4])) / (2 * distance(right_eye[0], right_eye[3]))

    return (left_ear + right_ear) / 2  # Average EAR
@app.route('/')
def home():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # Query to get attendance count by semester
    query = """
        SELECT u.semester, 
               COUNT(a.id) AS total_attendance, 
               (SELECT COUNT(*) FROM users WHERE semester = u.semester) AS total_students
        FROM users u
        LEFT JOIN attendance a ON u.id = a.user_id AND DATE(a.check_in_time) = CURDATE()
        GROUP BY u.semester
    """
    
    cursor.execute(query)
    semester_attendance = cursor.fetchall()

    conn.close()
    return render_template('index.html', semester_attendance=semester_attendance) 


# Validate Image Format
def is_valid_image(file_path):
    valid_formats = ["jpeg", "png", "bmp"]
    file_type = imghdr.what(file_path)
    return file_type in valid_formats

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        address = request.form['address']
        phone = request.form['phone']
        semester = request.form['semester']

        image_path = os.path.join(IMAGE_PATH, f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")

        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM users")
        existing_users = [row[0] for row in cursor.fetchall()]

        cap = cv2.VideoCapture(camera_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                flash("Error: Cannot access webcam!", "danger")
                return redirect(url_for('register'))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)
            mesh_results = face_mesh.process(rgb_frame)

            eye_open = False  # Flag to check if eyes are open

            if results.detections and mesh_results.multi_face_landmarks:
                for detection, landmarks in zip(results.detections, mesh_results.multi_face_landmarks):
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    # Ensure cropping values are within bounds
                    x, y = max(0, x), max(0, y)
                    w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

                    if w > 0 and h > 0:
                        face_img = frame[y:y+h, x:x+w]
                    else:
                        flash("Invalid face crop! Try again.", "danger")
                        continue

                    # Convert landmarks to pixel coordinates
                    landmark_coords = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in landmarks.landmark]

                    # Calculate EAR for open-eye detection
                    ear = calculate_ear(landmark_coords)
                    if ear > 0.25:  # Threshold for open eyes
                        eye_open = True

                    # Draw rectangle & instructions
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if eye_open:
                        cv2.putText(frame, "Press 'C' to Capture", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Open Your Eyes!", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Register Face", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and eye_open and results.detections:
                if face_img is not None and face_img.size > 0:
                    cv2.imwrite(image_path, face_img)  # Save cropped face

                    # ✅ Validate Image Format
                    if not is_valid_image(image_path):
                        os.remove(image_path)
                        flash("Invalid image format! Try again.", "danger")
                        continue

                    # ✅ Convert to JPEG
                    img = Image.open(image_path).convert("RGB")
                    img.save(image_path, "JPEG")

                else:
                    flash("No face detected! Try again.", "danger")
                    continue

                cap.release()
                cv2.destroyAllWindows()

                # ✅ Step 1: Check if face is already registered
                model_name = "ArcFace"  # More accurate than Facenet
                threshold = 0.5  # Lower threshold to prevent false matches

                for user_image in existing_users:
                    try:
                        result = DeepFace.verify(image_path, user_image, model_name=model_name, distance_metric="euclidean_l2")
                        print("Checking {image_path} vs {user_image}")
                        print(result)
                        if result["verified"] and result["distance"] < threshold:
                            flash("Face already registered!", "danger")
                            os.remove(image_path)  # Remove duplicate face
                            conn.close()
                            return redirect(url_for('register'))
                    except Exception as e:
                        print(f"DeepFace Error: {e}")
                        continue

                # ✅ Step 2: Save face details to the database
                cursor.execute("INSERT INTO users (name, image_path, email, address, phone, semester) VALUES (%s, %s, %s, %s, %s, %s)", 
                               (name, image_path, email, address, phone, semester))
                conn.commit()
                conn.close()

                flash("Face Registered Successfully!", "success")
                return redirect(url_for('register'))

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return render_template('register.html')
@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened():
            flash("Error: Cannot access webcam!", "danger")
            return redirect(url_for('recognize'))

        temp_path = "static/temp.jpg"
        face_detected = False
        eye_open = False

        while True:
            ret, frame = cap.read()
            if not ret:
                flash("Error: Cannot capture frame!", "danger")
                return redirect(url_for('recognize'))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    x, y = max(0, x), max(0, y)
                    w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

                    face_img = frame[y:y+h, x:x+w]
                    face_detected = True

                    # Detect facial landmarks and check eyes
                    landmark_coords = get_facial_landmarks(rgb_frame, detection)
                    # Convert landmarks to pixel coordinates
                    # landmark_coords = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in landmarks.landmark]

                    if not landmark_coords or len(landmark_coords) < max([33, 160, 158, 133, 153, 144, 362, 387, 385, 263, 373, 380]):  
                        print("⚠️ Warning: Not enough facial landmarks detected!")
                        continue  # Skip this frame if landmarks are incomplete
                    if landmark_coords:
                        ear = calculate_ear(landmark_coords)
                        eye_open = ear > 0.25  # Open eye threshold

                    # Draw rectangle around detected face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if eye_open:
                        cv2.putText(frame, "Press 'C' to Capture", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Open Your Eyes!", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Recognize Face", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and face_detected and eye_open:
                if face_img is not None and face_img.size > 0:
                    cv2.imwrite(temp_path, face_img)
                    print(f"✅ Image saved: {temp_path}")
                else:
                    flash("Error: No face detected. Try again.", "danger")
                    continue

                cap.release()
                cv2.destroyAllWindows()
                
                # Face Recognition Logic (Same as before)
                conn = connect_db()
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, image_path FROM users")
                users = cursor.fetchall()

                if not users:
                    flash("No registered users found!", "danger")
                    conn.close()
                    return render_template('recognize.html')

                models = ["Facenet", "ArcFace"]
                recognized_user = None

                threshold_facenet = 0.3  # Adjust for stricter matching
                threshold_arcface = 0.6  # Adjust for stricter matching

                for user_id, name, image_path in users:
                    if not os.path.exists(image_path):
                        continue

                    match_count = 0
                    for model in models:
                        try:
                            result = DeepFace.verify(temp_path, image_path, model_name=model, distance_metric="cosine")
                            print(f"🔍 Comparing: {temp_path} vs {image_path} using {model}")
                            print(result)
                            # if(model=="Facenet"):
                            #     if result['distance']>threshold_facenet:
                            #         print("changing facenet")
                            #         result["verified"]=False
                            # if(model=="ArcFace"):
                            #     if result['distance']>threshold_arcface:
                            #         print("changing ArcFace")
                            #         result["verified"]=False
                            if result["verified"]:
                                match_count += 1
                        except Exception as e:
                            print(f"❌ Error with model {model}: {e}")
                            match_count=0
                            continue

                    if match_count > 1:
                        recognized_user = (user_id, name)
                        break

                if recognized_user:
                    user_id, name = recognized_user
                    timestamp = datetime.now()
                    # cursor.execute("INSERT INTO attendance (user_id, timestamp) VALUES (%s, %s)", (user_id, timestamp))
                    # conn.commit()
                    # conn.close()

                    # Check if the user already checked in today
                    today_date = datetime.now().date()
                    cursor.execute(
                        "SELECT id, check_in_time, check_out_time FROM attendance WHERE user_id = %s AND DATE(check_in_time) = %s",
                        (user_id, today_date)
                    )
                    result = cursor.fetchone()

                    if result:
                        attendance_id, check_in, check_out = result

                        if check_out is None:  # If check-out is not marked yet
                            check_out_time = datetime.now()
                            cursor.execute("UPDATE attendance SET check_out_time = %s WHERE id = %s", (check_out_time, attendance_id))
                            conn.commit()
                            flash(f"User {name} checked out at {check_out_time}")
                        else:
                            flash(f"User {name} already checked out today.")

                    else:
                        check_in_time = datetime.now()
                        cursor.execute("INSERT INTO attendance (user_id, check_in_time) VALUES (%s, %s)", (user_id, check_in_time))
                        conn.commit()
                        flash(f"User {name} checked in at {check_in_time}")

                    conn.close()
                    # end chek
                    flash(f"✅ Attendance Marked for {name} at {timestamp}", "success")
                    return render_template('recognize.html', message=f"✅ Attendance Marked for {name} at {timestamp}", alert_type="success")

                conn.close()
                flash(f"❌ Face not recognized!", "danger")
                return render_template('recognize.html', message="❌ Face not recognized!", alert_type="danger")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return render_template('recognize.html')


@app.route('/attendance', methods=['GET'])
def attendance():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    search_query = request.args.get('search', '').strip()
    date_query = request.args.get('date', '').strip()
    if search_query or date_query:
        query = """
        SELECT a.id, u.name, u.email, u.semester, u.address, u.id AS user_id,
            a.check_in_time, a.check_out_time, 
            CONCAT(
                FLOOR(TIMESTAMPDIFF(MINUTE, a.check_in_time, a.check_out_time) / 60), 
                ' hours ', 
                MOD(TIMESTAMPDIFF(MINUTE, a.check_in_time, a.check_out_time), 60), 
                ' minutes'
            ) AS total_time
        FROM attendance a 
        JOIN users u ON a.user_id = u.id 
    """
    else:
        query = """
        SELECT a.id, u.name, u.email, u.semester, u.address, u.id AS user_id,
            a.check_in_time, a.check_out_time, 
            CONCAT(
                FLOOR(TIMESTAMPDIFF(MINUTE, a.check_in_time, a.check_out_time) / 60), 
                ' hours ', 
                MOD(TIMESTAMPDIFF(MINUTE, a.check_in_time, a.check_out_time), 60), 
                ' minutes'
            ) AS total_time
        FROM attendance a 
        JOIN users u ON a.user_id = u.id
        WHERE DATE(a.check_in_time) = CURDATE()
    """

    params = []

    if search_query:
        query += """
            AND (u.name LIKE %s OR u.email LIKE %s OR u.semester LIKE %s OR u.address LIKE %s)
        """
        search_pattern = '%' + search_query + '%'
        params.extend([search_pattern, search_pattern, search_pattern, search_pattern])

    if date_query:
        query += " AND DATE(a.timestamp) = %s"
        params.append(date_query)

    query += " ORDER BY a.timestamp DESC"
    cursor.execute(query, tuple(params))

    attendance_records = cursor.fetchall()
    total_count = len(attendance_records)
    conn.close()

    return render_template('attendance.html', attendance_records=attendance_records, total_count=total_count, search_query=search_query, date_query=date_query)


# student crud start
# ✅ View Student Details
@app.route('/student/<int:student_id>')
def student_detail(student_id):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (student_id,))
    student = cursor.fetchone()
    conn.close()
    return render_template('student_detail.html', student=student)

# ✅ Edit Student Page




# student crud end
def login_required(f):
    def wrap(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    
    wrap.__name__ = f.__name__
    return wrap

@app.route('/student/edit/<int:student_id>')
@login_required
def edit_student(student_id):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (student_id,))
    student = cursor.fetchone()
    conn.close()
    return render_template('edit_student.html', student=student)

# ✅ Update Student
@app.route('/student/update/<int:student_id>', methods=['POST'])
def update_student(student_id):
    name = request.form['name']
    email = request.form['email']
    semester = request.form['semester']
    address = request.form['address']

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE users 
        SET name=%s, email=%s, semester=%s, address=%s 
        WHERE id=%s
    """, (name, email, semester, address, student_id))
    conn.commit()
    conn.close()

    flash("Student details updated successfully!")
    return redirect(url_for('attendance'))
# login 
# 🟢 Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = connect_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM admin WHERE email=%s AND password=%s", (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session.permanent = False  # Make session expire when the browser closes
            session["user"] = user["email"]
            return redirect(url_for("attendance"))
        else:
            return flash(f"Invalid email or password!")
    
    return render_template("login.html")


# 🔴 Logout
@app.route("/logout")
def logout(): 
    session.pop("user", None)  # Remove user session
    session.modified = True  # Mark session as modified

    # Set cookie expiry to the past
    response = redirect(url_for("login"))
    response.set_cookie("session", "", expires=0)
    return response


# 🛡️ Authentication Check (Decorator)

if __name__ == '__main__':
    app.run(debug=True)

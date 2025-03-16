from deepface import DeepFace
import cv2
import os
# temp_path="static/temp.jpg"
# image_path="static/registered_faces\Bone.jpg"

# import cv2
# import mediapipe as mp
# from deepface import DeepFace

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# # Face Recognition Function
# def recognize_face(image_path, db_face_path, model="ArcFace"):
#     try:
#         # Load image
#         img = cv2.imread(image_path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Perform Face Recognition using DeepFace
#         result = DeepFace.verify(img1_path=image_path, img2_path=db_face_path, model_name=model, distance_metric="euclidean_l2", enforce_detection=False)
#         print(result)
#         # Check if face is verified
#         verified = result["verified"]
#         distance = result["distance"]
#         threshold = result["threshold"]

#         print(f"Match: {'✅ Yes' if verified else '❌ No'} (Distance: {distance:.4f} / Threshold: {threshold})")

#         # Detect Face Landmarks using MediaPipe
#         results = face_mesh.process(img_rgb)
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 for idx, landmark in enumerate(face_landmarks.landmark):
#                     h, w, _ = img.shape
#                     x, y = int(landmark.x * w), int(landmark.y * h)

#                     # Draw only key facial points (eyes, nose, mouth)
#                     if idx in [1, 4, 6, 9, 33, 61, 199, 263]:  # Key landmark indexes
#                         cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

#         # Show result
#         cv2.imshow("Face Recognition & Landmarks", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     except Exception as e:
#         print("❌ Error:", str(e))

# # Example usage
# recognize_face("static/registered_faces\Zin Maung Maung Phyo.jpg", "static/registered_faces\KyawZayarMin.jpg")





image_path="static/registered_faces\WaiLinNaing.jpg"
temp_path="static/registered_faces\Zin Maung Maung Phyo.jpg"
models = ["Facenet", "ArcFace"]
match_count = 0
for model in models:
    try:
        print(f"🔍 Comparing: {temp_path} vs {image_path} using {model}")
        print(f"Comparing: {temp_path} vs {image_path}")
        if not os.path.exists(temp_path) or not os.path.exists(image_path):
            print("Error: One of the image paths does not exist!")
        else:
            print("Both image paths exist!")
        result = DeepFace.verify(temp_path, image_path, model_name=model, distance_metric="cosine")
        print(result)
        if result["verified"]:
            match_count += 1
    except Exception as e:
        print(f"❌ Error with model {model}: {e}")
        continue

if match_count > 1:  # If 
    print("Match")
from deepface import DeepFace
import cv2
from PIL import Image
import os
image1 = "static/registered_faces/Wai.jpg"
image2 = "static/temp.jpg"
image_path="static/registered_faces\WaiLinNaing.jpg"
temp_path="static/registered_faces\KyawZayarMin.jpg"


models = ["Facenet", "ArcFace"]
recognized_user = None
match_count = 0
for model in models:
    try:
        print(f"🔍 Comparing: {temp_path} vs {image_path} using {model}")
        print(f"Comparing: {temp_path} vs {image_path}")
        if not os.path.exists(temp_path) or not os.path.exists(image_path):
            print("Error: One of the image paths does not exist!")
        else:
            print("Both image paths exist!")
        result = DeepFace.verify(temp_path, image_path, model_name=model, distance_metric="euclidean_l2")
        print(result)
        if result["verified"]:
            match_count += 1
    except Exception as e:
        print(f"❌ Error with model {model}: {e}")
        continue

if match_count > 0: 
    print("Image Match")
# try:
#     img = Image.open(image2)
#     img.show()  # Opens the image
#     print("✅ Successfully opened image with PIL")
# except Exception as e:
#     print(f"❌ PIL Error: {e}")
# # Convert images to RGB
# def convert_to_rgb(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"❌ OpenCV cannot read {image_path}")
#         return None
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     fixed_path = image_path.replace(".jpg", "_rgb.jpg")
#     cv2.imwrite(fixed_path, img)
#     return fixed_path

# fixed_image1 = convert_to_rgb(temp_path)
# fixed_image2 = convert_to_rgb(image_path)

# if fixed_image1 and fixed_image2:
#     try:
#         # result = DeepFace.verify(fixed_image1, fixed_image2, model_name="Facenet", distance_metric="euclidean_l2")
#         result = DeepFace.verify(temp_path, image_path, model_name=model, distance_metric="cosine")

#         print("✅ DeepFace Result:", result)
#     except Exception as e:
#         print(f"❌ DeepFace Error: {e}")

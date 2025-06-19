import cv2
import os

# Path to the root directory containing image folders
input_root = "/home/advay/Desktop/image_ml/input images"
output_root = "/home/advay/Desktop/image_ml/output"

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop through all subdirectories
for folder_name in os.listdir(input_root):
    input_folder_path = os.path.join(input_root, folder_name)
    
    if not os.path.isdir(input_folder_path):
        continue  # Skip non-directory files

    # Create corresponding output folder with "_face" suffix
    output_folder_name = folder_name + "_face"
    output_folder_path = os.path.join(output_root, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    # Loop through images in the current folder
    for filename in os.listdir(input_folder_path):
        input_image_path = os.path.join(input_folder_path, filename)
        
        # Read image
        img = cv2.imread(input_image_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Save each detected face
        for i, (x, y, w, h) in enumerate(faces):
            face_img = img[y:y+h, x:x+w]
            # Create a unique filename
            base_name = os.path.splitext(filename)[0]
            output_image_path = os.path.join(output_folder_path, f"{base_name}_face{i+1}.jpg")
            cv2.imwrite(output_image_path, face_img)

print("✅ Face extraction completed.")

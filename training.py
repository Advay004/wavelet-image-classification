import os
import cv2
import pywt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def wavelet_transform(img, mode='haar', level=1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray) / 255.0
    coeffs = pywt.wavedec2(img_gray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # Approximation set to zero
    img_wavelet = pywt.waverec2(coeffs_H, mode)
    img_wavelet = np.uint8(img_wavelet * 255)
    return img_wavelet

def prepare_data(input_root, image_size=(64, 64)):
    X = []
    y = []

    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label = int(folder_name)  # Assuming folders are named '0', '1', ...

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            img = cv2.imread(file_path)

            if img is None:
                continue

            # Resize original image
            img = cv2.resize(img, image_size)
            wavelet_img = wavelet_transform(img)

            # Resize wavelet image to match size
            wavelet_img = cv2.resize(wavelet_img, image_size)

            # Flatten both and stack
            combined = np.hstack((img.flatten(), wavelet_img.flatten()))
            X.append(combined)
            y.append(label)

    return np.array(X), np.array(y)

# Example usage
input_directory = "/home/advay/Desktop/image_ml/output"
X, y = prepare_data(input_directory)

print("✅ Data prepared.")
print("X shape:", X.shape)
print("y shape:", y.shape)


xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=0)
pipe=Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='rbf',C=10))])
pipe.fit(xtrain,ytrain)
print(pipe.score(xtest,ytest))

joblib.dump(pipe,'model.pkl')
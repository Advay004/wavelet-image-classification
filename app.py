from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import joblib
import os
import pywt

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class mapping
label_dict = {0: "Scarlett Johansson", 1: "Virat", 2: "Other"}

# Wavelet transform function
def wavelet_transform(img, mode='haar', level=1):
    """Apply wavelet transform to image"""
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = np.float32(img_gray) / 255.0
        coeffs = pywt.wavedec2(img_gray, mode, level=level)
        coeffs_H = list(coeffs)
        coeffs_H[0] *= 0  # Approximation set to zero
        img_wavelet = pywt.waverec2(coeffs_H, mode)
        img_wavelet = np.uint8(img_wavelet * 255)
        return img_wavelet
    except Exception as e:
        print(f"Error in wavelet transform: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None
    
    if request.method == "POST":
        try:
            # Check if model is loaded
            if model is None:
                error_message = "Model not loaded. Please check model.pkl file."
                return render_template("index.html", prediction=None, error=error_message)
            
            # Check if file was uploaded
            if 'image' not in request.files:
                error_message = "No file uploaded."
                return render_template("index.html", prediction=None, error=error_message)
            
            file = request.files["image"]
            
            # Check if file is selected
            if file.filename == '':
                error_message = "No file selected."
                return render_template("index.html", prediction=None, error=error_message)
            
            # Check file type
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
            if not ('.' in file.filename and 
                    file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                error_message = "Invalid file type. Please upload an image file."
                return render_template("index.html", prediction=None, error=error_message)
            
            # Process the image
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                error_message = "Could not decode image. Please try a different file."
                return render_template("index.html", prediction=None, error=error_message)

            # Resize and extract features
            img_resized = cv2.resize(img, (64, 64))
            wavelet_img = wavelet_transform(img_resized)
            
            if wavelet_img is None:
                error_message = "Error processing image."
                return render_template("index.html", prediction=None, error=error_message)
            
            wavelet_img = cv2.resize(wavelet_img, (64, 64))

            # Flatten and stack features
            features = np.hstack((img_resized.flatten(), wavelet_img.flatten())).reshape(1, -1)

            # Make prediction
            pred = model.predict(features)[0]
            prediction = label_dict.get(pred, "Unknown")
            
            print(f"Prediction made: {prediction}")

        except Exception as e:
            print(f"Error during prediction: {e}")
            error_message = f"An error occurred while processing the image: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error_message)

@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", 
                         prediction=None, 
                         error="File too large. Please upload an image smaller than 16MB."), 413

if __name__ == "__main__":
    print("Starting Face Classifier 3000...")
    print("Make sure model.pkl is in the same directory as this script.")
    app.run(debug=True, host='0.0.0.0', port=5000)
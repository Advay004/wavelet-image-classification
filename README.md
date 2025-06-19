
# Wavelet Face Classifier

A project that combines **image processing** and **machine learning** to classify face images into **Virat Kohli** or **Scarlett Johansson** using wavelet features and raw pixel features. It includes a **Flask web interface** for users to upload images and receive classification results.

## 🧠 What It Classifies
The model classifies uploaded face images into one of two categories:
- 🏏 **Virat Kohli**
- 🎬 **Scarlett Johansson**

## 🖼️ Inputs
- Users upload a face image (`.jpg`, `.png`, etc.) via a web form.
- The backend processes the image using OpenCV, applies wavelet transform, and performs classification.

## 📊 Model & Algorithm
- **Model Used:** `Support Vector Classifier (SVC)` from `scikit-learn`
- **Features:**
  - Flattened raw image (resized to 64x64)
  - Flattened wavelet-transformed image
- **Feature Shape:** Concatenated vector of original and wavelet features (e.g., `64x64x3 + 64x64`)
- **Face Detection:** Haarcascade via OpenCV
- **Wavelet Transform:** PyWavelets (`haar` basis)

## 🛠️ Tech Stack
- `Python 3.12+`
- `Flask` for backend
- `OpenCV` for image handling
- `scikit-learn` for training and inference
- `pywavelets` for wavelet feature extraction

## 🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Advay004/wavelet-image-classification.git
   cd wavelet-image-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser to access the web interface.

## 📈 Training Your Own Model
To train the model:
1. Organize your dataset into class-labeled folders (e.g., `data/0/`, `data/1/` for each class).
2. Run the training script:
   ```bash
   python training.py
   ```
This script:
- Loads images from the dataset
- Extracts features (resizing + wavelet transform)
- Trains the `Support Vector Classifier (SVC)`
- Saves the trained model to `model.pkl`

## 🔧 Future Improvements
- Improve face alignment and detection accuracy
- Add support for more classes using deep learning
- Enable real-time camera input
- Enhance web UI with better HTML/CSS styling
- Add Docker setup for easy deployment
- Deploy to a cloud platform (currently runs locally)

## 🧑‍💻 Author
Made with ❤️ by [Advay](https://github.com/Advay004)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

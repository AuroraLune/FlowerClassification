from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from skimage.feature import hog
import joblib
import webbrowser
from threading import Timer

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rf_model = joblib.load('model/rf_model.joblib')
cnn_model = load_model('model/cnn_model.h5')
label_encoder = joblib.load('model/label_encoder.joblib')
scaler = joblib.load('model/scaler.joblib')

def extract_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    hist_hue /= hist_hue.sum()
    hist_saturation /= hist_saturation.sum()
    hist_value /= hist_value.sum()

    return np.concatenate([hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()])

def extract_hog(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True)
    return fd

def predict_image_with_models(image):
    image_resized = cv2.resize(image, (224, 224))
    color_hist = extract_histogram(image_resized)
    hog_features = extract_hog(image_resized)
    combined_features = np.concatenate([color_hist, hog_features])
    combined_features = scaler.transform([combined_features])  

    rf_prediction = rf_model.predict(combined_features)
    rf_pred_class = label_encoder.inverse_transform(rf_prediction)[0]

    image_normalized = image_resized.astype('float32') / 255.0
    cnn_prediction = cnn_model.predict(image_normalized.reshape(1, 224, 224, 3))
    cnn_pred_class = label_encoder.inverse_transform([np.argmax(cnn_prediction)])[0]

    return rf_pred_class, cnn_pred_class

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            image = cv2.imread(image_path)
            rf_pred, cnn_pred = predict_image_with_models(image)

            return render_template('result.html', rf_pred=rf_pred, cnn_pred=cnn_pred, image_path=image_path)

    return render_template('index.html')

@app.route('/back', methods=['GET'])
def back():
    return redirect(url_for('index'))

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)

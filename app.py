from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import tensorflow as tf
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class_names = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 
    'melanoma', 'nevus', 'pigmented benign keratosis', 
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]

# Dictionary to hold models
models = {}

# Function to load a model and add it to the dictionary
def load_model_func(model_name, path):
    global models
    try:
        model = tf.keras.models.load_model(path)
        models[model_name] = model
        print(f"Model {model_name} loaded successfully from {path}.")
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")

load_model_func("skin_cancer_model_v6_9104", 'models/skin_cancer_model_v6_9104.keras')
model = models["skin_cancer_model_v6_9104"]

# Preprocessing the image
def preprocess_image(image_path):
    # Load the image with the specified target size
    img = load_img(image_path, target_size=(180, 180))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path, model, class_names):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    pred_class = class_names[pred_index]
    confidence = predictions[0][pred_index]
    print(f"Predictions: {predictions}")
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    return pred_class, confidence

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# Changed endpoint to '/analyze-image' to work with the React app
@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 401
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 402
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uploads_dir = 'uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        file_path = os.path.join(uploads_dir, filename)
        file.save(file_path)
        predicted_class, confidence = predict(file_path, model, class_names)
        os.remove(file_path)  # Optionally remove the file after analysis
        return jsonify({'prediction': predicted_class, 'confidence': f"{confidence:.2%}"})
    else:
        return jsonify({'error': 'File not allowed'}), 403
    
@app.route('/test')
def test():
    return "App is running!"

if __name__ == '__main__':
    app.run(debug=False)
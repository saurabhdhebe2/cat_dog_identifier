from flask import request, jsonify
from app import app
from app.utils import prepare_image
from tensorflow.keras.models import load_model
import os

model = load_model('models/cat_dog_classifier.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    img_array = prepare_image(img_path)

    prediction = model.predict(img_array)

    result = 'dog' if prediction[0][0] > 0.5 else 'cat'
    print(f"Prediction: {prediction[0][0]}, Type: {type(prediction[0][0])}")

    return jsonify({
        "prediction": result,
        "value":float(prediction[0][0])
        })



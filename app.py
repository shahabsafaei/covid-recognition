from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)
model = tf.keras.models.load_model('corona_detector_model.h5')

def preprocess_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_array, (150, 150))
    img_normalized = img_resized / 255.0
    return img_normalized.reshape(-1, 150, 150, 1)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)
        result = "COVID-19 Positive" if prediction[0][0] < 0.5 else "COVID-19 Negative"
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


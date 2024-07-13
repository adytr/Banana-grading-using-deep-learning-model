from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('ripeness.h5')

def predict_stage(image_path):
    size = (224, 224)
    image = Image.open(image_path)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.array(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        prediction = predict_stage(file_path)
        if np.argmax(prediction) == 0:
            result = "Unripe"
        elif np.argmax(prediction) == 1:
            result = "Overripe"
        else:
            result = "Ripe"
        os.remove(file_path)  # Clean up the uploaded file
        return render_template('result.html', result=result)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

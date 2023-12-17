from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
loaded_model = load_model('my_image_classifier_model.h5')

# Function to preprocess the image similar to your training data preprocessing
def preprocess_image(image):
    # Resize the image
    resized_image = cv2.resize(image, (64, 64))
    # Normalize pixel values
    processed_image = resized_image / 255.0
    return processed_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        # Read the uploaded image file
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image = preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        # Make prediction using the loaded model
        prediction = loaded_model.predict(processed_image)
        if prediction[0][0] > 0.5:
            result = 'FAKE'
        else:
            result = 'REAL'
        # Pass image data as base64 to result.html
        _, img_encoded = cv2.imencode('.jpg', image)
        img_data = img_encoded.tobytes()
        return render_template('result.html', image_data=img_data, prediction_result=result)

    # If it's a GET request, render the index.html template
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

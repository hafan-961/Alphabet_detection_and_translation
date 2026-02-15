from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
from src.engine import forward_propagation
from src.translator import WordTranslator
import os
app = Flask(__name__)
CORS(app)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, 'weights', 'model_params.npy')

# Load weights using the absolute path
parameters = np.load(weights_path, allow_pickle=True).item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_b64 = data.get('image')
    
    # Process base64 image
    encoded_data = image_b64.split(',')[1]
    nparr = np.frombuffer(base64.decodebytes(encoded_data.encode()), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess (Resize & Center)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        roi = thresh[y:y+h, x:x+w]
        side = max(w, h) + 40
        square = np.zeros((side, side), dtype="uint8")
        square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = roi
        resized = cv2.resize(square, (28, 28))
        
        x_input = resized.reshape(784, 1) / 255.0
        AL, _ = forward_propagation(x_input, parameters, 1.0, is_training=False)
        prediction = np.argmax(AL)
        char = chr(ord('A') + int(prediction))
        
        # Translate
        translator.current_word = char
        spanish = translator.translate_to_spanish()
        
        return jsonify({'letter': char, 'translation': spanish})
    return jsonify({'error': 'No input'}), 400

if __name__ == "__main__":
    app.run()
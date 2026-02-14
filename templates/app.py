from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from src.engine import forward_propagation
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app) 

# 1.load your scratch-built model
parameters = np.load("weights/model_params.npy", allow_pickle=True).item()

def preprocess_image(image_data):
    #convert base64 string from phone to OpenCV image
    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    #standardize (The same logic as your predict_image.py)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        roi = thresh[y:y+h, x:x+w]
        size = max(w, h) + 20
        final = np.zeros((size, size), dtype="uint8")
        final[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = roi
        resized = cv2.resize(final, (28, 28))
        return resized.reshape(784, 1) / 255.0
    return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_b64 = data.get('image')
    
    x_input = preprocess_image(image_b64)
    if x_input is not None:
        #run numPy forward prop
        AL, _ = forward_propagation(x_input, parameters, 1.0, is_training=False)
        prediction = int(np.argmax(AL))
        char = chr(ord('A') + prediction)
        
        #translate
        spanish = GoogleTranslator(source='en', target='es').translate(char)
        
        return jsonify({'letter': char, 'translation': spanish})
    return jsonify({'error': 'No letter detected'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
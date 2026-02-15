import os, base64, cv2, numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.engine import forward_propagation
from deep_translator import GoogleTranslator
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

# Load Weights
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
params = np.load(os.path.join(BASE_DIR, 'weights', 'model_params.npy'), allow_pickle=True).item()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image'].split(",")[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Find letters
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
        
        word = ""
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h > 10:
                roi = img[y:y+h, x:x+w]
                size = max(w, h) + 40
                sq = np.zeros((size, size), dtype="uint8")
                sq[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = roi
                # Predict
                x_in = cv2.resize(sq, (28, 28)).reshape(784, 1) / 255.0
                AL, _ = forward_propagation(x_in, params, 1.0, False)
                word += chr(ord('A') + np.argmax(AL))
        
        # Spell Check individual word
        corrected = str(TextBlob(word).correct()).upper() if word else ""
        return jsonify({'english': corrected})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    text = request.json.get('text', '')
    try:
        spanish = GoogleTranslator(source='en', target='es').translate(text)
        return jsonify({'spanish': spanish})
    except: return jsonify({'spanish': "Error"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
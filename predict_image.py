import cv2
import numpy as np
from src.engine import forward_propagation
from deep_translator import GoogleTranslator
from textblob import TextBlob  # source

# tuned parameters
IMAGE_PATH = "paper.jpg"
SPACE_THRESHOLD_FACTOR = 1.8    #adjusted for your handwriting
MIN_LETTER_HEIGHT = 25 
MIN_LETTER_WIDTH = 8

print("--- Step 1: Loading ---", flush=True)
parameters = np.load("weights/model_params.npy", allow_pickle=True).item()
img = cv2.imread(IMAGE_PATH)
if img is None: exit("Image not found")

#preprocessing 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)

#sgmentation 
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

raw_text = ""
prev_x_end = None

for i, ctr in enumerate(contours):
    x, y, w, h = cv2.boundingRect(ctr)
    if h > MIN_LETTER_HEIGHT and w > MIN_LETTER_WIDTH:
        # SPACE LOGIC
        if prev_x_end is not None:
            gap = x - prev_x_end
            if gap > (w * SPACE_THRESHOLD_FACTOR):
                raw_text += " "
        
        #PREDICTION
        roi = thresh[y:y+h, x:x+w]
        size = max(w, h) + 20
        final_image = np.zeros((size, size), dtype="uint8")
        final_image[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = roi
        final_image = cv2.resize(final_image, (28, 28))
        
        x_input = final_image.reshape(784, 1) / 255.0
        AL, _ = forward_propagation(x_input, parameters, 1.0, is_training=False)
        
        char = chr(ord('A') + np.argmax(AL))
        raw_text += char
        prev_x_end = x + w

# language correction 
corrected_text = str(TextBlob(raw_text).correct()).upper()

#output and tranlation
print("\n" + "="*40)
print(f"RAW DETECTION:       {raw_text}")
print(f"LANGUAGE CORRECTED:  {corrected_text}")

if corrected_text.strip():
    try:
        translator = GoogleTranslator(source='en', target='es')
        spanish = translator.translate(corrected_text)
        print(f"SPANISH TRANSLATION: {spanish}")
    except: pass
print("="*40)
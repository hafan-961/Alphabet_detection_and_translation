import cv2
import numpy as np
from src.engine import forward_propagation
from src.translator import WordTranslator

params = np.load("weights/model_params.npy", allow_pickle=True).item()
translator = WordTranslator()
canvas = np.zeros((400, 400), dtype="uint8")
drawing = False

def draw(event, x, y, flags, param):
    global drawing, canvas
    if event == cv2.EVENT_LBUTTONDOWN: drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing: cv2.circle(canvas, (x,y), 15, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP: drawing = False

cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas", draw)

while True:
    cv2.imshow("Canvas", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): canvas.fill(0)
    elif key == ord('t'):
        print(f"Spanish: {translator.translate_to_spanish()}")
        translator.reset()
    elif key == ord('a'):
        coords = cv2.findNonZero(canvas)
        if coords is not None:
            x,y,w,h = cv2.boundingRect(coords)
            roi = canvas[y:y+h, x:x+w]
            # Center in 28x28
            side = max(w, h) + 40
            square = np.zeros((side, side), dtype="uint8")
            square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = roi
            x_inp = cv2.resize(square, (28, 28)).reshape(784, 1) / 255.0
            al, _ = forward_propagation(x_inp, params, 1.0, False)
            char = translator.append_letter(np.argmax(al))
            print(f"Detected: {char} | Word: {translator.current_word}")
            canvas.fill(0)

cv2.destroyAllWindows()
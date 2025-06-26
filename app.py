# app.py
import os
import csv
import traceback
from datetime import datetime

import cv2
import base64
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# ─── Logging setup ─────────────────────────────────────────────────────────────

def record(message: str, level: str = 'INFO'):
    print(message)
record("Logging initialized.")

# ─── Class mapping ─────────────────────────────────────────────────────────────
mapping = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

# ─── Flask & Socket.IO init ────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'secret!')
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True
)
record("Flask and SocketIO initialized.")

# ─── Load Haar cascade ─────────────────────────────────────────────────────────
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    record(f"Could not load Haar cascade from {cascade_path}", level='ERROR')
    raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
record("Haar cascade loaded.")

# ─── Load model ────────────────────────────────────────────────────────────────
NUM_CLASSES = 7
model = models.efficientnet_b0(pretrained=False)
in_feats = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_feats, NUM_CLASSES)

try:
    sd = torch.load("model(V2).pth", map_location=torch.device('cpu'))
    print("Checkpoint keys:", list(sd.keys())[:5])
    model.load_state_dict(sd)
    model.eval()
    record("Model(V2).pth loaded into EfficientNet-B0 and set to eval mode.")
except Exception as e:
    record(f"Failed to load model(V2).pth: {e}", level='ERROR')
    raise

# ─── Preprocessing pipeline ───────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),                         # ← convert numpy.ndarray → PIL Image
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
record("Transform pipeline ready.")

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ─── Socket.IO Frame Handler ──────────────────────────────────────────────────
@socketio.on('frame')
def handle_frame(data):

    results = []
    try:
        # Decode the base64 frame into a CV2 image
        _, b64 = data.split(',', 1)
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Decoded frame is None")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            
            roi = gray[y:y+h, x:x+w]                  # numpy.ndarray
            tensor = transform(roi).unsqueeze(0)      # now works without TypeError

            with torch.no_grad():
                
                out   = model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                pred  = int(probs.argmax().item())
                conf  = float(probs[pred].item())

            results.append({
                'box':        [int(x), int(y), int(w), int(h)],
                'pred_label': mapping[pred],
                'confidence': round(conf, 4)
            })
            record(f"Face at {(x,y,w,h)} -> {mapping[pred]} ({conf:.4f})")

        emit('predictions', results)
        if results:
            record(f"Emitted {len(results)} prediction(s)")

    except Exception as e:
        tb = traceback.format_exc()
        record(f"Error in handle_frame: {e}\n{tb}", level='ERROR')
        emit('error', {'message': str(e)})

# ─── Server Startup ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )

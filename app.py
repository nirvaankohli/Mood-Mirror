# app.py
import os
import csv
import traceback
from datetime import datetime

import cv2
import base64
import numpy as np
import torch
from torchvision import transforms
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from model import CustomResNet

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging_csv_path = 'logging.csv'
if not os.path.exists(logging_csv_path):
    with open(logging_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "level", "message"])

def record(message: str, level: str = 'INFO'):
    """Print and append a log message to CSV."""
    ts = datetime.now().isoformat()
    log_line = f"[{level}] {message}"
    print(log_line)
    with open(logging_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ts, level, message])

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
model = CustomResNet(num_classes=7)
try:
    sd = torch.load("model(V1).pth", map_location=torch.device('cpu'))
    
    print([k for k in sd.keys()][:5])

    model.load_state_dict(sd)
    model.eval()
    record("Model loaded and set to eval mode.")
except Exception as e:
    record(f"Failed to load model: {e}", level='ERROR')
    raise

# ─── Preprocessing pipeline ───────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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
        # Split off the "data:image/…;base64," prefix
        _, b64 = data.split(',', 1)
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Decoded frame is None")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            tensor = transform(roi).unsqueeze(0)  # [1, C, H, W]

            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                pred = int(probs.argmax().item())
                conf = float(probs[pred].item())

            results.append({
                'box':       [int(x), int(y), int(w), int(h)],
                'pred_label': mapping[pred],
                'confidence': round(conf, 4)
            })
            record(f"Face at {(x,y,w,h)} -> {mapping[pred]} ({conf:.4f})")

        emit('predictions', results)
        if len(results) != 0:
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

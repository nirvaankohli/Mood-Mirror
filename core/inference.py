import os
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

class EmotionModel:
    EMOTIONS = [
        "angry", "disgust", "fear",
        "happy", "neutral", "sad", "surprise"
    ]

    def __init__(
        self,
        onnx_path: str = "models/model(V4).onnx",
        dnn_prototxt="models/deploy.prototxt",
        dnn_weights="models/res10_300x300_ssd_iter_140000.caffemodel"

    ):
        # --- ONNX runtime setup ---
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # --- DNN face detector setup ---
        if dnn_prototxt and dnn_weights:
            self.net = cv2.dnn.readNetFromCaffe(dnn_prototxt, dnn_weights)
        else:
            # fallback to Haar
            cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_file)

        # --- Preprocessing pipeline ---
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ])

    def _detect_faces_dnn(self, frame: np.ndarray, conf_threshold=0.5):
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < conf_threshold:

                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def predict(self, frame: np.ndarray) -> str:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # choose detector
        if hasattr(self, 'net'):
            faces = self._detect_faces_dnn(frame)
        else:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

        if not faces:
            return "no_face_detected"

        # pick largest face
        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
        face_rgb = frame[y:y+h, x:x+w]

        # preprocess & run ONNX
        img = Image.fromarray(face_rgb)
        x_in = self.preprocess(img).unsqueeze(0).numpy().astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: x_in})
        probs = outputs[0][0]
        idx = int(np.argmax(probs))
        return self.EMOTIONS[idx]

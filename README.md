# Realtime-Emotion-Detection-CNN

Mood-Mirror/                       :contentReference
├── src/
│   └── moodmirror/ ← Python desktop app
|       |               
|       ├── assets/
│       │   ├── Satoshi-Variable.ttf
│       │   ├── Satoshi-VariableItalic.ttf
|       |   ├── icons /
|       |   |   └── favicon.ico
│       │   └── images/
|       |       ├── logo.png
|       |       ├── logo.svg
|       |       ├── Mood Mirror Logo.png
│       │       └── calm_scenery.png
|       |
|       ├── core/
│       │   ├── __init__.py
│       │   ├── notifications.py
|       |   └── inference.py
|       ├── data/
│       │   ├── __init__.py
│       │   ├── db.db
│       │   └── username.txt
|       ├── db/
│       │   ├── __init__.py
│       │   └── api.py
│       ├── models/               ← Pretrained models :contentReference
│       │   ├── model_v4.onnx
│       │   ├── deploy.prototxt
│       │   └── res10_300x300_ssd_iter_140000.caffemodel
│       ├── ui/                   ← QML UI files :contentReference
│       │   ├── Dashboard.qml
│       │   └── WelcomePage.qml
│       ├── __init__.py
│       └── main.py               ← Application entrypoint 
├── .gitignore
├── README.md
├── poetry.lock
├── pyproject.toml

<div align="center">
  <a href="https://shipwrecked.hackclub.com/?t=ghrm" target="_blank">
    <img src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/739361f1d440b17fc9e2f74e49fc185d86cbec14_badge.png" 
         alt="This project is part of Shipwrecked, the world's first hackathon on an island!" 
         style="width: 35%;">
  </a>
</div>
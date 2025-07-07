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

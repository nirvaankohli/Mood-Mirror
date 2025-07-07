#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from datetime import date, timedelta

import cv2
import PySide6
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtCore import (

    Qt, 
    QDir, 
    QObject, 
    Slot, 
    Signal,
    QUrl, 
    QAbstractListModel, 
    QModelIndex, Property, QTimer

)

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel,
    QPushButton, QVBoxLayout, QWidget, QMessageBox
)

from moodmirror.core.inference import EmotionModel
from moodmirror.db.api import Sessions, Events

import sys, os
from PySide6.QtCore import QDir

if getattr(sys, "frozen", False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(__file__)

ui_path = os.path.join(BASE_PATH, "ui")
QDir.addSearchPath("ui", ui_path)

icons_path = os.path.join(BASE_PATH, "assets", "icons")
QDir.addSearchPath("icons", icons_path)

# ─── 1. QQuick style & Qt DLL path ────────────────────────────────────────
QQuickStyle.setStyle("Basic")

pyside_dir = Path(PySide6.__file__).parent
os.environ["PATH"] = str(pyside_dir) + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(pyside_dir))

# ─── 2. Resource path helper ─────────────────────────────────────────────
def resource_path(*parts: str) -> Path:
    """
    Absolute path to bundled resources, handling development and PyInstaller.
    """
    base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent
    return base.joinpath(*parts)

# ─── 3. Define key directories ────────────────────────────────────────────
USERNAME_FILE = resource_path("data", "username.txt")
UI_DIR        = resource_path("ui")
ASSETS_DIR    = resource_path("assets")
MODELS_DIR    = resource_path("models")

# ─── 4. QML search paths ──────────────────────────────────────────────────
QDir.addSearchPath("ui",     str(UI_DIR))
QDir.addSearchPath("assets", str(ASSETS_DIR))
QDir.addSearchPath("models", str(MODELS_DIR))

# ─── 5. Stress data model ─────────────────────────────────────────────────
class StressEntry:
    def __init__(self, date_str: str, score: float):
        self.date = date_str
        self.score = score

class StressModel(QAbstractListModel):

    DateRole  = Qt.UserRole + 1
    ScoreRole = Qt.UserRole + 2

    def __init__(self, parent=None):

        super().__init__(parent)

        self._entries = []
        self.load_dummy_last_7_days()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._entries)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        entry = self._entries[index.row()]
        if role == StressModel.DateRole:
            return entry.date
        if role == StressModel.ScoreRole:
            return entry.score
        return None

    def roleNames(self):
        return {
            StressModel.DateRole:  b"date",
            StressModel.ScoreRole: b"score",
        }

    def load_dummy_last_7_days(self):
        today = date.today()
        self.beginResetModel()
        self._entries = [
            StressEntry(
                (today - timedelta(days=6 - i)).strftime("%b %d"),
                (i * 13 + 25) % 100
            ) for i in range(7)
        ]
        self.endResetModel()

# ─── 6. Backend for userName binding ─────────────────────────────────────
class Backend(QObject):
    userNameChanged = Signal()

    def __init__(self):
        super().__init__()
        try:
            self._userName = USERNAME_FILE.read_text().strip()
        except FileNotFoundError:
            self._userName = ""

    @Property(str, notify=userNameChanged)
    def userName(self) -> str:
        return self._userName

    @Slot(str)
    def saveUser(self, name: str):
        self._userName = name
        USERNAME_FILE.write_text(name)
        self.userNameChanged.emit()

# ─── 7. Main camera + inference window ──────────────────────────────────
class MoodMirrorWindow(QMainWindow):

    def __init__(self, model: EmotionModel):
        # keep session object around but don’t start it yet
        self.session = Sessions()
        self.events  = Events()

        self.weights = {
            
            "angry": 1, 
            
            "disgust": .3, 
            
            "fear": .7,
        
            "happy": 0, 
            
            "neutral": .2, 
            
            "sad": 1, 
            
            "surprise": "no effect"

        }

        super().__init__()
        self.setWindowTitle("Mood Mirror")

        self.model        = model
        self.current_frame = None
        self.call_active   = False
        self.paused        = False

        # ——— video display ———
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        # rounded corners + black border
        self.video_label.setStyleSheet(
            "border:2px solid black; "
            "border-radius:15px;"
        )

        
        self.call_button = QPushButton("Call")
        self.call_button.setCheckable(True)
        self.call_button.toggled.connect(self.toggle_call)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self.toggle_pause)

        self.start_button = QPushButton("Start Session")
        self.start_button.setCheckable(True)
        self.start_button.setStyleSheet("background-color:red; color:white;")
        self.start_button.toggled.connect(self.toggle_session)

        container = QWidget()
        main_layout = QVBoxLayout(container)

        row = QVBoxLayout()
        row.addWidget(self.call_button)
        row.addWidget(self.video_label, 1)  
        row.addWidget(self.pause_button)
        main_layout.addLayout(row)

        main_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.setCentralWidget(container)

        self.cap   = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

        self.inference_timer = QTimer(self)
        self.inference_timer.setInterval(500)
        self.inference_timer.timeout.connect(self.run_inference)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # convert & detect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if hasattr(self.model, "net"):
            faces = self.model._detect_faces_dnn(rgb, conf_threshold=0.5)
        else:
            gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = self.model.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))

        # draw boxes on a copy
        vis = rgb.copy()
        for x, y, w, h in faces:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255,0,0), 2)

        # keep the un-blurred frame for inference
        self.current_frame = vis

        # apply light blur if on a “call”
        display = vis
        if self.call_active:
            display = cv2.GaussianBlur(vis, (15,15), 0)

        # show it
        h, w, ch = display.shape
        img = QImage(display.data, w, h, ch*w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def toggle_session(self, active: bool):

        if active:

            self.session_id = self.session.create_session()
            self.inference_timer.start()
            self.start_button.setText("Stop Session")

            self.start_button.setStyleSheet("background-color:green; color:white;")
        
        else:

            self.inference_timer.stop()

            self.session.close_session(self.session_id, None, self.events.get_max_number_in_session(self.session_id)[7])

            self.close()

            if hasattr(self, "_welcome_engine"):
                for obj in self._welcome_engine.rootObjects():
                    obj.setProperty("visible", False)

            self._dash_engine = QQmlApplicationEngine()
            dash_ctxt = self._dash_engine.rootContext()
            dash_ctxt.setContextProperty("stressModel", self.stressModel)
            dash_ctxt.setContextProperty("controller",   self)
            self._dash_engine.load(QUrl("ui:Dashboard.qml"))

    def toggle_pause(self, paused: bool):

        self.paused = paused

        if paused:

            self.inference_timer.stop()
            self.pause_button.setText("Resume")

        else:

            self.inference_timer.start()
            self.pause_button.setText("Pause")

            


    def toggle_call(self, calling: bool):

        self.call_active = calling

        if calling:
            self.call_button.setText("End Call")
        else:
            self.call_button.setText("Call")

    def run_inference(self):

        if self.current_frame is None:

            return
        
        label = self.model.predict(self.current_frame)

        if self.call_active == True:

            event_type = "call"

        else:

            event_type = "regular"

        if label != "surprise" and label != "no_face_detected":

            self.events.report_event(self.session_id, label, event_type, weight=self.weights[label])


    def closeEvent(self, event):

        self.cap.release()

        super().closeEvent(event)


# ─── 8. QML controller ───────────────────────────────────────────────────
class AppController(QObject):
    def __init__(self, engine: QQmlApplicationEngine):

        super().__init__()

        self._welcome_engine = engine
        self.model = EmotionModel()
        self.stressModel = StressModel()
        self._dash_engine = None
        self._inference = None

    @Slot()

    def continueToApp(self):

        print("Got Button Click")           # <-- should appear in console
        # hide welcome…
        for obj in self._welcome_engine.rootObjects():
            obj.setProperty("visible", False)
        # load dashboard…
        self._dash_engine = QQmlApplicationEngine()
        dash_ctxt = self._dash_engine.rootContext()
        dash_ctxt.setContextProperty("stressModel", self.stressModel)
        dash_ctxt.setContextProperty("controller",   self)
        self._dash_engine.load("ui:Dashboard.qml")

    @Slot()
    def startWorkSession(self):
        win = MoodMirrorWindow(self.model)
        win.resize(800, 600)
        win.show()
        self._inference = win

# ─── 9. Entry point ─────────────────────────────────────────────────────
def main():

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    from PySide6.QtGui     import QIcon

    
    app.setWindowIcon(QIcon("icons:favicon.ico"))


    backend = Backend()
    controller = AppController(engine)

    ctxt = engine.rootContext()
    ctxt.setContextProperty("backend",    backend)
    ctxt.setContextProperty("controller", controller)

    engine.load("ui:WelcomePage.qml")



    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

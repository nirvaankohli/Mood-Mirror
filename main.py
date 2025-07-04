import sys, os, datetime, cv2

from PySide6.QtQuickControls2 import QQuickStyle

QQuickStyle.setStyle("Basic")

from PySide6.QtWidgets import (

    QApplication, 

    QMainWindow, 

    QLabel,

    QPushButton, 

    QVBoxLayout, 
    
    QWidget, 

    QMessageBox
    
)

from PySide6.QtCore    import (

    QObject, 

    Slot, 
    
    Qt, 

    QUrl,

    QAbstractListModel, 

    QModelIndex, 
    
    Property, 
    
    Signal
)

from PySide6.QtQml     import QQmlApplicationEngine
from PySide6.QtGui     import (
    
    QImage, 
    QPixmap)

from core.inference import EmotionModel

# ─── Ensure Qt DLLs are on the PATH ───────────────────────────────────────
import PySide6
pyside_pkg_dir = os.path.dirname(PySide6.__file__)
os.environ["PATH"] = pyside_pkg_dir + os.pathsep + os.environ.get("PATH", "")
try:    os.add_dll_directory(pyside_pkg_dir)
except: pass

# ─── StressModel: 7-day bar data ─────────────────────────────────────────
class StressEntry:
    def __init__(self, date: str, score: float):
        self._date = date
        self._score = score

class StressModel(QAbstractListModel):
    DateRole  = Qt.UserRole + 1
    ScoreRole = Qt.UserRole + 2

    def __init__(self, entries=None):
        super().__init__()
        self._entries = entries or []

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._entries)

    def data(self, index: QModelIndex, role: int = ...) -> object:
        e = self._entries[index.row()]
        if role == StressModel.DateRole:  return e._date
        if role == StressModel.ScoreRole: return e._score
        return None

    def roleNames(self) -> dict:
        return {
            StressModel.DateRole:  b"date",
            StressModel.ScoreRole: b"score",
        }

    def load_dummy_last_7_days(self):
        today = datetime.date.today()
        self.beginResetModel()
        self._entries = []
        for i in range(7):
            d = today - datetime.timedelta(days=6 - i)
            score = (i * 13 + 25) % 100
            self._entries.append(StressEntry(d.strftime("%b %d"), score))
        self.endResetModel()

# ─── Backend: bindable userName ──────────────────────────────────────────
class Backend(QObject):
    userNameChanged = Signal()

    def __init__(self):
        super().__init__()
        self._userName = ""
        try:
            with open("username.txt", "r") as f:
                self._userName = f.read().strip()
        except FileNotFoundError:
            pass

    @Property(str, notify=userNameChanged)
    def userName(self):
        return self._userName

    @Slot(str)
    def saveUser(self, name):
        self._userName = name
        with open("username.txt", "w") as f:
            f.write(name)
        self.userNameChanged.emit()

# ─── Old camera+inference window ────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, model: EmotionModel):
        super().__init__()
        self.setWindowTitle("Mood Mirror")
        self.model = model

        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.btn_check   = QPushButton("Force Check")
        self.btn_check.clicked.connect(self.run_inference_on_current_frame)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.btn_check)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap   = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = self.startTimer(30)
        self.current_frame = None

    def timerEvent(self, event):
        ret, frame = self.cap.read()
        if not ret: return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if hasattr(self.model, "net"):
            faces = self.model._detect_faces_dnn(rgb, conf_threshold=0.5)
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = self.model.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50)
            )

        for (x,y,w,h) in faces:
            cv2.rectangle(rgb, (x,y), (x+w,y+h), (255,0,0), 2)

        self.current_frame = rgb
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def run_inference_on_current_frame(self):
        if not self.current_frame: return
        label = self.model.predict(self.current_frame)
        QMessageBox.information(self, "Mood Mirror", f"Detected emotion: {label}")

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

# ─── Controller: hides welcome, loads dashboard, spawns inference ───────
class AppController(QObject):
    def __init__(self, engine: QQmlApplicationEngine):
        super().__init__()
        self._welcome_engine = engine

        # prepare DNN-based EmotionModel
        self.model = EmotionModel(
            onnx_path   = "models/model(V4).onnx",
            dnn_prototxt= "models/deploy.prototxt",
            dnn_weights = "models/res10_300x300_ssd_iter_140000.caffemodel"
        )

        # prepare stress data
        self.stressModel = StressModel()
        self.stressModel.load_dummy_last_7_days()

        self._dash_engine    = None
        self._inference_win  = None

    @Slot()
    def continueToApp(self):
        # hide welcome window
        for obj in self._welcome_engine.rootObjects():
            obj.setProperty("visible", False)

        # new engine for dashboard
        self._dash_engine = QQmlApplicationEngine()
        ctxt = self._dash_engine.rootContext()
        ctxt.setContextProperty("stressModel", self.stressModel)
        ctxt.setContextProperty("controller",   self)

        dash_qml = os.path.abspath(os.path.join("ui", "Dashboard.qml"))
        print(f"Loading dashboard QML from: {dash_qml}")
        self._dash_engine.load(QUrl.fromLocalFile(dash_qml))

        if not self._dash_engine.rootObjects():
            print("❌ ERROR: Dashboard.qml failed to load!")

    @Slot()
    def startWorkSession(self):
        win = MainWindow(self.model)
        win.resize(800, 600)
        win.show()
        self._inference_win = win

def main():
    app    = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    backend    = Backend()
    controller = AppController(engine)

    ctxt = engine.rootContext()
    ctxt.setContextProperty("backend",    backend)
    ctxt.setContextProperty("controller", controller)

    welcome = os.path.abspath(os.path.join("ui", "WelcomePage.qml"))
    engine.load(QUrl.fromLocalFile(welcome))
    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

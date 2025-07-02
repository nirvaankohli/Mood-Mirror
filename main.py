import sys
import os
import cv2
import numpy as np
import PySide6
from PySide6.QtQuickControls2 import QQuickStyle
QQuickStyle.setStyle("Basic")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel,
    QPushButton, QVBoxLayout, QWidget, QMessageBox
)
from PySide6.QtCore    import QObject, Slot, QTimer, Qt, QUrl, QCoreApplication, Property
from PySide6.QtGui     import QImage, QPixmap
from PySide6.QtQml     import QQmlApplicationEngine

from core.inference import EmotionModel

# ─── Ensure Qt DLLs are on the PATH ───────────────────────────────────────

pyside_pkg_dir = os.path.dirname(PySide6.__file__)
os.environ["PATH"] = pyside_pkg_dir + os.pathsep + os.environ.get("PATH", "")

try:
    os.add_dll_directory(pyside_pkg_dir)
    print(f"🔧 Registered {pyside_pkg_dir} for DLL loading")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────

class Backend(QObject):
    def __init__(self):
        super().__init__()
        self._userName = ""
        try:
            with open("username.txt", "r") as file:
                self._userName = file.read().strip()
        except FileNotFoundError:
            pass

    @Property(str, constant=False)
    def userName(self):
        return self._userName

    @Slot(str)
    def saveUser(self, name):
        self._userName = name
        with open("username.txt", "w") as file:
            file.write(name)
        try:
            self.userNameChanged.emit()
        except:
            pass

class MainWindow(QMainWindow):
    def __init__(self, model: EmotionModel):
        super().__init__()
        self.setWindowTitle("Mood Mirror")
        self.model = model

        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.btn_check = QPushButton("Force Check")
        self.btn_check.clicked.connect(self.run_inference_on_current_frame)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.btn_check)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = QTimer(self, interval=30)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()
        self.current_frame = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces via DNN if available, else Haar
        if hasattr(self.model, "net"):
            faces = self.model._detect_faces_dnn(rgb, conf_threshold=0.5)
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = self.model.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

        self.current_frame = rgb
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def run_inference_on_current_frame(self):
        if self.current_frame is None:
            return
        label = self.model.predict(self.current_frame)
        QMessageBox.information(self, "Mood Mirror", f"Detected emotion: {label}")

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

class AppController(QObject):
    def __init__(self, engine: QQmlApplicationEngine):
        super().__init__()
        self.engine = engine
        self.main_window = None

        # Load EmotionModel with DNN SSD face detector
        self.model = EmotionModel(
            onnx_path="models/model(V4).onnx",
            dnn_prototxt="models/deploy.prototxt",
            dnn_weights="models/res10_300x300_ssd_iter_140000.caffemodel"
        )

    @Slot()
    def continueToApp(self):
        # Hide QML window
        for obj in self.engine.rootObjects():
            obj.setProperty('visible', False)
        # Launch main widget UI
        self.main_window = MainWindow(self.model)
        self.main_window.resize(800, 600)
        self.main_window.show()

def main():
    app = QApplication(sys.argv)

    # Set up QML engine
    engine = QQmlApplicationEngine()
    qml_dir = os.path.join(os.path.dirname(PySide6.__file__), 'qml')
    plugins_dir = os.path.join(os.path.dirname(PySide6.__file__), 'plugins')
    engine.addImportPath(qml_dir)
    QCoreApplication.setLibraryPaths([plugins_dir])

    controller = AppController(engine)
    backend = Backend()

    engine.rootContext().setContextProperty("controller", controller)
    engine.rootContext().setContextProperty("backend", backend)

    qml_path = os.path.abspath(os.path.join("ui", "WelcomePage.qml"))
    engine.load(QUrl.fromLocalFile(qml_path))
    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

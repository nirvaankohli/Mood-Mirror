#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from datetime import date, timedelta

import cv2
import PySide6

from PySide6.QtCore import (
    Qt, QSize, QPropertyAnimation, QEasingCurve,
    Property, Signal, QTimer, QUrl,
    QDir, QObject, Slot, QAbstractListModel, QModelIndex
)
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QImage
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFrame, QHBoxLayout, QVBoxLayout, QWidget
)
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from moodmirror.core.inference import EmotionModel
from moodmirror.db.api import Sessions, Events

# ─── 1. Determine BASE_PATH & register search paths ─────────────────────
if getattr(sys, "frozen", False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(__file__)

# QML/UI search paths
ui_dir = os.path.join(BASE_PATH, "ui")
assets_dir = os.path.join(BASE_PATH, "assets")
models_dir = os.path.join(BASE_PATH, "models")
icons_dir = os.path.join(assets_dir, "icons")

QDir.addSearchPath("ui",     ui_dir)
QDir.addSearchPath("assets", assets_dir)
QDir.addSearchPath("models", models_dir)
QDir.addSearchPath("icons",  icons_dir)

# ─── 2. QQuick style & DLL path ──────────────────────────────────────────
QQuickStyle.setStyle("Basic")

pyside_dir = Path(PySide6.__file__).parent
os.environ["PATH"] = str(pyside_dir) + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(pyside_dir))

# ─── 3. Resource‐path helper ─────────────────────────────────────────────
def resource_path(*parts: str) -> Path:
    base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent
    return base.joinpath(*parts)

USERNAME_FILE = resource_path("data", "username.txt")


# ─── 4. Stress data model ────────────────────────────────────────────────
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
        self.sessions = Sessions()
        self.events = Events()
        self._time_range_days = 7
        self._metric = "Stress score"
        self.load_from_db()

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

    @Slot(int, str)
    def reload(self, time_range_index: int, metric: str):
        # Map index to days
        days_map = [7, 30, 90, 180, 10000]  # 10000 = all time
        self._time_range_days = days_map[time_range_index]
        self._metric = metric
        self.load_from_db()

    def load_from_db(self):
        # Get sessions in range
        if self._time_range_days >= 10000:
            sessions = self.sessions.get_all_sessions()
        else:
            sessions = self.sessions.get_sessions_in_last_x_days(self._time_range_days)
        # Aggregate by day
        day_map = {}
        for sess in sessions:
            # session: (session_id, day, start_time, end_time, overall_stress_score, total_reminders)
            day = sess[1]
            if day not in day_map:
                day_map[day] = {"stress": [], "reminders": 0}
            if sess[4] is not None:
                day_map[day]["stress"].append(sess[4])
            if sess[5] is not None:
                day_map[day]["reminders"] += int(sess[5])
        # Build entries for each day in range
        days = sorted(day_map.keys())
        self.beginResetModel()
        self._entries = []
        for d in days:
            if self._metric == "Stress score":
                vals = day_map[d]["stress"]
                score = sum(vals)/len(vals) if vals else 0
            else:
                score = day_map[d]["reminders"]
            # Format date as 'Mon DD'
            try:
                date_str = date.fromisoformat(d).strftime("%b %d")
            except Exception:
                date_str = d
            self._entries.append(StressEntry(date_str, score))
        self.endResetModel()

    @Slot(int, result='QVariant')
    def get(self, index):
        if 0 <= index < len(self._entries):
            entry = self._entries[index]
            return {"date": entry.date, "score": entry.score}
        return None


# ─── 5. Backend for userName binding ─────────────────────────────────────
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


# ─── 6. AnimatedIconButton ───────────────────────────────────────────────
class AnimatedIconButton(QPushButton):
    bgColorChanged     = Signal()
    borderColorChanged = Signal()
    scaleChanged       = Signal()

    def __init__(
        self,
        icon_off_path: str,
        icon_on_path: str,
        base_size: QSize,
        icon_size: QSize,
        off_bg: QColor,
        off_border: QColor,
        on_bg: QColor,
        on_border: QColor,
        parent=None
    ):
        super().__init__(parent)

        # store geometry parameters
        self.base_size = base_size
        self.icon_size = icon_size
        self._margin   = 10   # px padding so scaling never crops
        total = QSize(
            base_size.width()  + 2*self._margin,
            base_size.height() + 2*self._margin
        )
        self.setFixedSize(total)

        # load white icons
        self.icon_off = self._make_white_icon(icon_off_path, icon_size)
        self.icon_on  = self._make_white_icon(icon_on_path,  icon_size)
        self.setIcon(self.icon_off)
        self.setIconSize(self.icon_size)

        # initial colors & scale
        self.off_bg,      self.off_border   = off_bg,     off_border
        self.on_bg,       self.on_border    = on_bg,      on_border
        self._bgColor     = self.off_bg
        self._borderColor = self.off_border
        self._scale       = 1.0
        self._max_scale   = 1.2

        # appearance & state
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self.toggled.connect(self._on_toggled)

        # animations
        self.anim_scale  = QPropertyAnimation(self, b"scale",       self)
        self.anim_bg     = QPropertyAnimation(self, b"bgColor",     self)
        self.anim_border = QPropertyAnimation(self, b"borderColor", self)
        for anim in (self.anim_scale, self.anim_bg, self.anim_border):
            anim.setDuration(500)
            anim.setEasingCurve(QEasingCurve.InOutQuad)

    def _make_white_icon(self, svg_path: str, size: QSize) -> QIcon:
        svg  = QSvgRenderer(svg_path)
        base = QPixmap(size)
        base.fill(Qt.transparent)
        p    = QPainter(base)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        svg.render(p)
        p.end()
        mask  = base.createMaskFromColor(Qt.transparent)
        white = QPixmap(size)
        white.fill(Qt.white)
        white.setMask(mask)
        return QIcon(white)

    def enterEvent(self, event):
        self.anim_scale.stop()
        self.anim_scale.setStartValue(self._scale)
        self.anim_scale.setEndValue(self._max_scale)
        self.anim_scale.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.anim_scale.stop()
        self.anim_scale.setStartValue(self._scale)
        self.anim_scale.setEndValue(1.0)
        self.anim_scale.start()
        super().leaveEvent(event)

    def _on_toggled(self, checked: bool):
        # swap icon
        # bg color animation
        self.anim_bg.stop()
        self.anim_bg.setStartValue(self._bgColor)
        self.anim_bg.setEndValue(self.on_bg if checked else self.off_bg)
        self.anim_bg.start()
        
        # border color animation
        self.anim_border.stop()
        self.anim_border.setStartValue(self._borderColor)
        self.anim_border.setEndValue(self.on_border if checked else self.off_border)
        self.anim_border.start()

        self.setIcon(self.icon_on if checked else self.icon_off)


    @Property(float, notify=scaleChanged)
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, s: float):
        self._scale = s
        self.scaleChanged.emit()
        self.update()

    @Property(QColor, notify=bgColorChanged)
    def bgColor(self) -> QColor:
        return self._bgColor

    @bgColor.setter
    def bgColor(self, c: QColor):
        self._bgColor = c
        self.bgColorChanged.emit()
        self.update()

    @Property(QColor, notify=borderColorChanged)
    def borderColor(self) -> QColor:
        return self._borderColor

    @borderColor.setter
    def borderColor(self, c: QColor):
        self._borderColor = c
        self.borderColorChanged.emit()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        W, H = self.width(), self.height()
        cx, cy = W/2, H/2

        # draw scaled circle
        bw, bh = self.base_size.width(), self.base_size.height()
        rw, rh = bw*self._scale, bh*self._scale
        x0, y0 = cx - rw/2, cy - rh/2

        pen = QPen(self._borderColor)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(self._bgColor)
        painter.drawEllipse(int(x0), int(y0), int(rw), int(rh))

        # draw scaled icon
        pix = self.icon().pixmap(self.icon_size)
        if not pix.isNull():
            iw, ih = pix.width()*self._scale, pix.height()*self._scale
            pix = pix.scaled(int(iw), int(ih), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            px = cx - pix.width()/2
            py = cy - pix.height()/2
            painter.drawPixmap(int(px), int(py), pix)

        painter.end()


# ─── 7. Main camera + inference window ─────────────────────────────────
class MoodMirrorWindow(QMainWindow):
    def __init__(self, model: EmotionModel, app_controller, parent=None):
        super().__init__(parent)           
        self._app_controller = app_controller
        self.setWindowTitle("Mood Mirror")
        self.model   = model
        self.session = Sessions()
        self.events  = Events()
        self.weights = {
            "angry": 1, "disgust": .3, "fear": .7,
            "happy": 0, "neutral": .2, "sad": 1,
            "surprise": "no effect"
        }

        self.call_active   = False
        self.paused        = False
        self.current_frame = None

        # icon target sizes
        self.icon_size_small = QSize(32, 32)
        self.icon_size_large = QSize(40, 40)

        # ── Video display ───────────────────────────────────────
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: black;
            border: 2px solid #000;
            border-radius: 15px;
        """)

        # ── Buttons ─────────────────────────────────────────────
        small_btn = QSize(60, 60)
        large_btn = QSize(80, 80)

        self.call_button = AnimatedIconButton(
            "icons:call_start.svg", "icons:call_end.svg",
            small_btn, self.icon_size_small,
            QColor(0,0,0,0), QColor(255,255,255),
            QColor(231,76,60), QColor(255,255,255),
            parent=self
        )
        self.call_button.toggled.connect(self.toggle_call)

        self.record_button = AnimatedIconButton(
            "icons:video.svg", "icons:video_start.svg",
            large_btn, self.icon_size_large,
            QColor(0,0,0,0), QColor(255,255,255),
            QColor(231,76,60), QColor(255,255,255),
            parent=self
        )
        self.record_button.toggled.connect(self.toggle_record)

        self.pause_button = AnimatedIconButton(
            "icons:pause.svg", "icons:resume.svg",
            small_btn, self.icon_size_small,
            QColor(0,0,0,0), QColor(255,255,255),
            QColor(231,76,60), QColor(255,255,255),
            parent=self
        )
        self.pause_button.toggled.connect(self.toggle_pause)

        # ── Bottom panel ────────────────────────────────────────
        bottom = QFrame()
        bottom.setStyleSheet("background-color: #2f2f2f;")
        layout = QHBoxLayout(bottom)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(20)  # +10px space between buttons
        layout.addStretch(1)
        layout.addWidget(self.call_button)
        layout.addWidget(self.record_button)
        layout.addWidget(self.pause_button)
        layout.addStretch(1)

        # ── Main layout ─────────────────────────────────────────
        container   = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addWidget(bottom,         stretch=0)
        self.setCentralWidget(container)

        # ── Video capture & timers ──────────────────────────────
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(30)

        self.inference_timer = QTimer(self)
        self.inference_timer.setInterval(100)
        self.inference_timer.timeout.connect(self.run_inference)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if hasattr(self.model, "net"):
            faces = self.model._detect_faces_dnn(rgb, conf_threshold=0.5)
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = self.model.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))

        vis = rgb.copy()
        for x, y, w, h in faces:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255,0,0), 2)
        self.current_frame = vis

        display = cv2.GaussianBlur(vis, (15,15), 0) if self.call_active else vis
        h, w, ch = display.shape
        img = QImage(display.data, w, h, ch*w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def toggle_call(self, on: bool):
        self.call_active = on

    def toggle_pause(self, paused: bool):
        self.paused = paused
        if paused:
            self.inference_timer.stop()
        elif self.record_button.isChecked():
            self.inference_timer.start()

    def toggle_record(self, recording: bool):

        if recording:

            self.session_id = self.session.create_session()
            self.inference_timer.start()

        else:
        
            self.inference_timer.stop()

            row = self.events.get_max_number_in_session(self.session_id)
            
            stress_score, stress_reminder = row[7], row[8]

            self.session.close_session(stress_reminder, self.session_id, None, stress_score)
            self._switch_to_dashboard()

    def run_inference(self):

        if self.current_frame is None:
            return
        label = self.model.predict(self.current_frame)
        evt   = "call" if self.call_active else "regular"

        print("frame_recorded")

        if label not in ("surprise", "no_face_detected"):
            self.events.report_event(
                self.session_id, label, evt,
                weight=self.weights[label]
            )

    def _switch_to_dashboard(self):
        self.close()
        if hasattr(self, "_welcome_engine"):
            for o in self._welcome_engine.rootObjects():
                o.setProperty("visible", False)
        dash = QQmlApplicationEngine()
        ctxt = dash.rootContext()
        ctxt.setContextProperty("stressModel", getattr(self, "stressModel", None))
        ctxt.setContextProperty("controller",   self)
        dash.load(QUrl("ui:Dashboard.qml"))
        self._dash_engine = dash

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


# ─── 8. QML controller ───────────────────────────────────────────────────
class AppController(QObject):
    def __init__(self, engine: QQmlApplicationEngine):
        super().__init__()
        self._welcome_engine = engine
        self.model          = EmotionModel()
        self.stressModel    = StressModel()
        self._dash_engine   = None
        self._inference     = None

    @Slot()
    def continueToApp(self):
        for obj in self._welcome_engine.rootObjects():
            obj.setProperty("visible", False)
        dash = QQmlApplicationEngine()
        ctxt = dash.rootContext()
        ctxt.setContextProperty("stressModel", self.stressModel)
        ctxt.setContextProperty("controller",   self)
        dash.load("ui:Dashboard.qml")
        self._dash_engine = dash

    @Slot()
    def startWorkSession(self):
        win = MoodMirrorWindow(self.model, app_controller=self)
        win.resize(800, 600)
        win.show()
        self._inference = win


# ─── 9. Entry point ─────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icons:favicon.ico"))

    backend    = Backend()
    controller = AppController(QQmlApplicationEngine())

    ctxt = controller._welcome_engine.rootContext()
    ctxt.setContextProperty("backend",    backend)
    ctxt.setContextProperty("controller", controller)
    controller._welcome_engine.load("ui:WelcomePage.qml")

    if not controller._welcome_engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

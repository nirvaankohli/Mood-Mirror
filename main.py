import sys, os
import PySide6
from PySide6.QtCore    import QUrl, QCoreApplication
from PySide6.QtWidgets import QApplication
from PySide6.QtQml     import QQmlApplicationEngine

def main():
    print("🏁 main() start")
    app = QApplication(sys.argv)

    # ─── Detect Qt6*.dll directory ─────────────────────────────────────────────
    pkg_dir = os.path.dirname(PySide6.__file__)
    qt_bin_dir = None

    # Walk the PySide6 install tree looking for Qt6Core.dll
    for root, dirs, files in os.walk(pkg_dir):
        if any(f.lower() == "qt6core.dll" for f in files):
            qt_bin_dir = root
            break

    if qt_bin_dir:
        print(f"🔧 Adding DLL directory: {qt_bin_dir}")
        os.add_dll_directory(qt_bin_dir)
    else:
        print(f"⚠️  Could not locate Qt6Core.dll under {pkg_dir}")
    # ──────────────────────────────────────────────────────────────────────────

    engine = QQmlApplicationEngine()

    # QML import plugins (QtQuick, Controls…)
    qml_dir = os.path.join(pkg_dir, "qml")
    print(f"🔧 Adding QML import path: {qml_dir}")
    engine.addImportPath(qml_dir)

    # Qt plugin subfolders (platforms, QtQuick/Controls…)
    plugins_dir = os.path.join(pkg_dir, "plugins")
    print(f"🔧 Setting Qt plugin paths: {plugins_dir}")
    QCoreApplication.setLibraryPaths([plugins_dir])

    # Load your QML UI
    qml_file = os.path.abspath(os.path.join("ui", "MainWindow.qml"))
    print(f"🔍 Loading QML from: {qml_file}")
    engine.load(QUrl.fromLocalFile(qml_file))

    if not engine.rootObjects():
        print("❌ QML load failed — check qml_dir, plugins_dir, and imports")
        sys.exit(1)

    print("✅ QML loaded, entering event loop")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

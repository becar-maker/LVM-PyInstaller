import sys, cv2, numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from roi_label import VideoLabel
from eulerian import TemporalIIRBandpass

APP_TITLE = "Live Video Magnification (Python)"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1100, 700)

        # --- video prikaz
        self.label = VideoLabel()
        self.label.setMinimumSize(640, 360)
        self.label.roiChanged.connect(self.on_roi_changed)
        self.setCentralWidget(self.label)

        # --- desni panel z nastavitvami
        dock = QtWidgets.QDockWidget("Controls", self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)

        # gumbi
        self.btn_open = QtWidgets.QPushButton("Open Video")
        self.btn_play = QtWidgets.QPushButton("Play"); self.btn_play.setCheckable(True)
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.btn_open); hb.addWidget(self.btn_play)
        form.addRow(hb)

        # fps in pas
        self.spin_fps  = QtWidgets.QDoubleSpinBox(); self.spin_fps.setRange(1,1000); self.spin_fps.setValue(30); self.spin_fps.setSuffix(" fps")
        self.spin_low  = QtWidgets.QDoubleSpinBox(); self.spin_low.setRange(0.01,200); self.spin_low.setValue(2.0); self.spin_low.setSuffix(" Hz")
        self.spin_high = QtWidgets.QDoubleSpinBox(); self.spin_high.setRange(0.05,300); self.spin_high.setValue(8.0); self.spin_high.setSuffix(" Hz")
        form.addRow("FPS (override):", self.spin_fps)
        form.addRow("Low cut (Hz):",   self.spin_low)
        form.addRow("High cut (Hz):",  self.spin_high)

        # ojačanje
        self.slider_amp = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider_amp.setRange(0,400); self.slider_amp.setValue(100)
        self.lbl_amp = QtWidgets.QLabel("1.00×")
        amp_row = QtWidgets.QHBoxLayout(); amp_row.addWidget(self.slider_amp); amp_row.addWidget(self.lbl_amp)
        form.addRow("Amplification:", amp_row)

        # info o ROI
        self.lbl_roi = QtWidgets.QLabel("ROI: full frame"); form.addRow(self.lbl_roi)
        dock.setWidget(w)

        # povezave
        self.btn_open.clicked.connect(self.open_video)
        self.btn_play.toggled.connect(self.toggle_play)
        self.slider_amp.valueChanged.connect(lambda v: self.lbl_amp.setText(f"{v/100:.2f}×"))

        # stanje
        self.cap = None
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.next_frame)
        self.iir = None
        self.roi = None  # (x,y,w,h)

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "", "Video files (*.mp4 *.avi *.mkv *.mov *.mpg *.mpeg);;All files (*)")
        if not path: return
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open video."); return

        cam_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        if cam_fps > 0:
            try: self.spin_fps.setValue(float(cam_fps))
            except: pass

        self.iir = None
        self.roi = None
        self.label.clear_roi()
        self.btn_play.setChecked(False)
        self.next_frame(draw_only=True)

    def toggle_play(self, playing: bool):
        if not self.cap:
            self.btn_play.setChecked(False); return
        if playing:
            fps = float(self.spin_fps.value())
            self.timer.start(int(1000 / max(fps, 1)))
            self.btn_play.setText("Pause")
        else:
            self.timer.stop()
            self.btn_play.setText("Play")

    def on_roi_changed(self, rect: QtCore.QRect):
        if rect is None:
            self.roi = None
            self.lbl_roi.setText("ROI: full frame")
            self.iir = None
        else:
            self.roi = (rect.x(), rect.y(), rect.width(), rect.height())
            self.lbl_roi.setText(f"ROI: x={rect.x()} y={rect.y()} w={rect.width()} h={rect.height()}")
            self.iir = None  # nova ROI → reset filtra

    def next_frame(self, draw_only=False):
        if not self.cap: return
        ok, frame = self.cap.read()
        if not ok:
            self.btn_play.setChecked(False); return

        disp = frame.copy()
        x, y, w, h = (0,0,frame.shape[1],frame.shape[0]) if not self.roi else self.roi
        roi_view = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

        if self.iir is None:
            fps  = float(self.spin_fps.value())
            low  = float(self.spin_low.value())
            high = float(self.spin_high.value())
            high = min(high, fps/2 - 0.01)
            if high <= low: high = low + 0.01
            self.spin_high.setValue(high)
            self.iir = TemporalIIRBandpass(low, high, fps, shape=gray.shape)

        band = self.iir.update(gray)
        amp = self.slider_amp.value()/100.0
        out = np.clip(gray + amp*band, 0.0, 1.0)

        out_bgr = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        disp[y:y+h, x:x+w] = out_bgr

        self.show_frame(disp)
        if draw_only: self.btn_play.setChecked(False)

    def show_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


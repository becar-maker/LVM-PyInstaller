import sys, cv2, numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from roi_label import VideoLabel
from riesz import RieszMotionMagnifier

APP_TITLE = "Live Video Magnification (Riesz Motion)"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1100, 700)

        # video view
        self.label = VideoLabel()
        self.label.setMinimumSize(640, 360)
        self.label.roiChanged.connect(self.on_roi_changed)
        self.setCentralWidget(self.label)

        # controls dock
        dock = QtWidgets.QDockWidget("Controls", self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)

        # open / play
        self.btn_open = QtWidgets.QPushButton("Open Video")
        self.btn_play = QtWidgets.QPushButton("Play"); self.btn_play.setCheckable(True)
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.btn_open); hb.addWidget(self.btn_play)
        form.addRow(hb)

        # fps + band (Hz)
        self.spin_fps  = QtWidgets.QDoubleSpinBox(); self.spin_fps.setRange(1,1000); self.spin_fps.setValue(30); self.spin_fps.setSuffix(" fps")
        self.spin_low  = QtWidgets.QDoubleSpinBox(); self.spin_low.setRange(0.01,200); self.spin_low.setValue(2.0); self.spin_low.setSuffix(" Hz")
        self.spin_high = QtWidgets.QDoubleSpinBox(); self.spin_high.setRange(0.05,300); self.spin_high.setValue(8.0); self.spin_high.setSuffix(" Hz")
        form.addRow("FPS (override):", self.spin_fps)
        form.addRow("Low cut (Hz):",   self.spin_low)
        form.addRow("High cut (Hz):",  self.spin_high)

        # Riesz pyramid levels
        self.spin_levels = QtWidgets.QSpinBox(); self.spin_levels.setRange(1,5); self.spin_levels.setValue(3)
        form.addRow("Levels:", self.spin_levels)

        # amplification – up to 100× (0.01× step)
        self.slider_amp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_amp.setRange(0, 10000)      # 0 … 100.00 ×
        self.slider_amp.setValue(100)           # default 1.00 ×
        self.lbl_amp = QtWidgets.QLabel("1.00×")
        amp_row = QtWidgets.QHBoxLayout(); amp_row.addWidget(self.slider_amp); amp_row.addWidget(self.lbl_amp)
        form.addRow("Amplification:", amp_row)

        # loop + roi info
        self.chk_loop = QtWidgets.QCheckBox("Loop at end"); self.chk_loop.setChecked(True)
        form.addRow(self.chk_loop)
        self.lbl_roi = QtWidgets.QLabel("ROI: full frame"); form.addRow(self.lbl_roi)

        dock.setWidget(w)
        self.setStatusBar(QtWidgets.QStatusBar(self))

        # signals
        self.btn_open.clicked.connect(self.open_video)
        self.btn_play.toggled.connect(self.toggle_play)
        self.slider_amp.valueChanged.connect(lambda v: self.lbl_amp.setText(f"{v/100:.2f}×"))
        # sprememba parametrov -> ponastavi magnifier ob naslednjem kadru
        self.spin_fps.valueChanged.connect(self.invalidate_magnifier)
        self.spin_low.valueChanged.connect(self.invalidate_magnifier)
        self.spin_high.valueChanged.connect(self.invalidate_magnifier)
        self.spin_levels.valueChanged.connect(self.invalidate_magnifier)

        # state
        self.cap = None
        self.frame_count = 0
        self.frame_idx = 0
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.next_frame)
        self.magnifier = None
        self.roi = None  # (x,y,w,h)

    def invalidate_magnifier(self):
        self.magnifier = None

    # --- open video
    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Video files (*.mp4 *.avi *.mkv *.mov *.mpg *.mpeg);;All files (*)")
        if not path:
            return

        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open video.")
            return

        cam_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        if cam_fps > 0:
            try: self.spin_fps.setValue(float(cam_fps))
            except: pass
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_idx = 0

        ok, first = self.cap.read()
        if ok:
            self.show_frame(first)
        # rewind so Play starts from frame 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0

        self.magnifier = None
        self.roi = None
        self.label.clear_roi()
        self.btn_play.setChecked(False)
        self.update_status()

    # --- play/pause
    def toggle_play(self, playing: bool):
        if not self.cap:
            self.btn_play.setChecked(False)
            return
        if playing:
            fps = float(self.spin_fps.value())
            self.timer.start(max(1, int(1000 / max(fps, 1))))
            self.btn_play.setText("Pause")
        else:
            self.timer.stop()
            self.btn_play.setText("Play")

    # --- ROI changed
    def on_roi_changed(self, rect: QtCore.QRect):
        if rect is None:
            self.roi = None
            self.statusBar().showMessage("ROI cleared")
            self.lbl_roi.setText("ROI: full frame")
            self.magnifier = None
        else:
            self.roi = (rect.x(), rect.y(), rect.width(), rect.height())
            self.lbl_roi.setText(f"ROI: x={rect.x()} y={rect.y()} w={rect.width()} h={rect.height()}")
            self.magnifier = None  # reinit for new ROI

    # --- frame loop
    def next_frame(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            if self.chk_loop.isChecked():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                return
            else:
                self.btn_play.setChecked(False)
                return

        self.frame_idx += 1

        disp = frame.copy()
        x, y, w, h = (0, 0, frame.shape[1], frame.shape[0]) if not self.roi else self.roi

        roi_view = frame[y:y+h, x:x+w]
        # ker imaš črno-beli vir: če je 1-kanalno, vzemi direktno; sicer pretvori v gray
        if roi_view.ndim == 2:
            gray = roi_view.astype(np.float32) / 255.0
        else:
            # tudi če je “B/W v 3 kanalih”, to samo izbere svetlost
            gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # inicializacija Riesz magnifier-ja
        if self.magnifier is None:
            fps  = float(self.spin_fps.value())
            low  = float(self.spin_low.value())
            high = float(self.spin_high.value())
            high = min(high, fps/2 - 0.01)
            if high <= low: high = low + 0.01
            self.spin_high.setValue(high)
            levels = int(self.spin_levels.value())
            self.magnifier = RieszMotionMagnifier(levels, low, high, fps, shape_hw=gray.shape)

        alpha = self.slider_amp.value() / 100.0   # 0 … 100.00
        out = self.magnifier.magnify(gray, alpha)

        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        if roi_view.ndim == 2:
            out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR)
        else:
            out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR)
        disp[y:y+h, x:x+w] = out_bgr

        self.show_frame(disp)
        self.update_status()

    def update_status(self):
        if self.frame_count > 0:
            self.statusBar().showMessage(f"Frame {self.frame_idx}/{self.frame_count}")
        else:
            self.statusBar().showMessage(f"Frame {self.frame_idx}")

    def show_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

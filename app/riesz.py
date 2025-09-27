import sys, os, shutil, cv2, numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from roi_label import VideoLabel
from riesz import RieszMotionMagnifier

APP_TITLE = "Live Video Magnification (Riesz Motion)"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 780)

        # === CENTRALNI LAYOUT: VIDEO ZGORAJ, ORODJA SPODAJ ===
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(10)

        # --- video view ---
        self.label = VideoLabel()
        self.label.setMinimumSize(720, 405)
        self.label.roiChanged.connect(self.on_roi_changed)
        vbox.addWidget(self.label, stretch=1)

        # --- Playback vrstica (Open/Transform/Play/Stop/Save/Export MP4) ---
        row1 = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Open")
        self.btn_transform = QtWidgets.QPushButton("Transform (Riesz)")
        self.btn_play = QtWidgets.QPushButton("Play"); self.btn_play.setCheckable(True)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_save = QtWidgets.QPushButton("Save As…")
        self.btn_export = QtWidgets.QPushButton("Export MP4…")
        for b in (self.btn_open, self.btn_transform, self.btn_play, self.btn_stop, self.btn_save, self.btn_export):
            b.setMinimumWidth(110)
        row1.addWidget(self.btn_open)
        row1.addWidget(self.btn_transform)
        row1.addStretch(1)
        row1.addWidget(self.btn_play)
        row1.addWidget(self.btn_stop)
        row1.addWidget(self.btn_save)
        row1.addWidget(self.btn_export)
        vbox.addLayout(row1)

        # --- Timeline + Slow speed ---
        row2 = QtWidgets.QHBoxLayout()
        self.slider_timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider_timeline.setRange(0, 0)
        self.lbl_time = QtWidgets.QLabel("00:00 / 00:00")
        row2.addWidget(QtWidgets.QLabel("Timeline:"))
        row2.addWidget(self.slider_timeline, stretch=1)
        row2.addWidget(self.lbl_time)
        row2.addSpacing(20)
        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_speed.setRange(1, 100)      # 1% … 100%
        self.slider_speed.setValue(100)         # default 100%
        self.lbl_speed = QtWidgets.QLabel("Speed: 100%")
        row2.addWidget(QtWidgets.QLabel("Slow speed:"))
        row2.addWidget(self.slider_speed)
        row2.addWidget(self.lbl_speed)
        vbox.addLayout(row2)

        # --- Parametri (FPS, pas, levels, amplification, loop) ---
        grid = QtWidgets.QGridLayout()
        r = 0
        self.spin_fps  = QtWidgets.QDoubleSpinBox(); self.spin_fps.setRange(1,1000); self.spin_fps.setValue(30); self.spin_fps.setSuffix(" fps")
        self.spin_low  = QtWidgets.QDoubleSpinBox(); self.spin_low.setRange(0.01,200); self.spin_low.setValue(2.0); self.spin_low.setSuffix(" Hz")
        self.spin_high = QtWidgets.QDoubleSpinBox(); self.spin_high.setRange(0.05,300); self.spin_high.setValue(8.0); self.spin_high.setSuffix(" Hz")
        self.spin_levels = QtWidgets.QSpinBox(); self.spin_levels.setRange(1,5); self.spin_levels.setValue(3)

        # Amplification slider: 1 … 100 (label prikazuje ×1 … ×100)
        self.slider_amp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_amp.setRange(1, 100)
        self.slider_amp.setValue(50)
        self.lbl_amp = QtWidgets.QLabel(f"{self.slider_amp.value()}×")
        amp_row = QtWidgets.QHBoxLayout(); amp_row.addWidget(self.slider_amp); amp_row.addWidget(self.lbl_amp)

        self.chk_loop = QtWidgets.QCheckBox("Loop at end"); self.chk_loop.setChecked(True)

        grid.addWidget(QtWidgets.QLabel("FPS (override):"), r, 0); grid.addWidget(self.spin_fps, r, 1); r+=1
        grid.addWidget(QtWidgets.QLabel("Low cut (Hz):"),   r, 0); grid.addWidget(self.spin_low, r, 1); r+=1
        grid.addWidget(QtWidgets.QLabel("High cut (Hz):"),  r, 0); grid.addWidget(self.spin_high, r, 1); r+=1
        grid.addWidget(QtWidgets.QLabel("Levels:"),         r, 0); grid.addWidget(self.spin_levels, r, 1); r+=1
        grid.addWidget(QtWidgets.QLabel("Amplification:"),  r, 0); grid.addLayout(amp_row, r, 1); r+=1
        grid.addWidget(self.chk_loop, r, 0, 1, 2); r+=1

        grid_box = QtWidgets.QGroupBox("Parameters")
        grid_box.setLayout(grid)
        vbox.addWidget(grid_box)

        self.setCentralWidget(central)

        # --- Status bar ---
        self.setStatusBar(QtWidgets.QStatusBar(self))

        # === SIGNALI ===
        self.btn_open.clicked.connect(self.open_video)
        self.btn_transform.clicked.connect(self.transform_video)
        self.btn_play.toggled.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_save.clicked.connect(self.save_as)
        self.btn_export.clicked.connect(self.export_mp4)

        self.slider_amp.valueChanged.connect(lambda v: self.lbl_amp.setText(f"{v}×"))
        self.slider_speed.valueChanged.connect(self.on_speed_changed)

        for w in (self.spin_fps, self.spin_low, self.spin_high, self.spin_levels):
            w.valueChanged.connect(self.invalidate_processed)

        self.slider_timeline.sliderPressed.connect(self.pause_for_seek)
        self.slider_timeline.sliderReleased.connect(self.seek_to_slider)

        # === STANJE ===
        self.src_path = None
        self.src_cap = None
        self.src_fps = 30.0
        self.src_frame_count = 0

        self.proc_path = None
        self.play_cap = None
        self.play_frame_count = 0
        self.play_frame_idx = 0

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.next_frame_play)

        self.roi = None  # (x,y,w,h)

    # --------- helpers ----------
    def on_speed_changed(self, v: int):
        self.lbl_speed.setText(f"Speed: {v}%")
        if self.btn_play.isChecked():
            fps = float(self.spin_fps.value())
            speed = max(1, int(self.slider_speed.value())) / 100.0
            interval_ms = max(1, int(1000 / max(fps * speed, 0.01)))
            self.timer.start(interval_ms)

    def invalidate_processed(self):
        self.proc_path = None
        if self.play_cap: self.play_cap.release(); self.play_cap = None
        self.slider_timeline.setRange(0, 0)
        self.lbl_time.setText("00:00 / 00:00")
        self.statusBar().showMessage("Parameters changed — please Transform again.")

    def format_time(self, frames, fps):
        if fps <= 0: return "00:00"
        total_sec = int(frames / fps)
        m, s = divmod(total_sec, 60)
        return f"{m:02d}:{s:02d}"

    def pause_for_seek(self):
        if self.btn_play.isChecked():
            self.btn_play.setChecked(False)

    def seek_to_slider(self):
        if not self.play_cap: return
        idx = int(self.slider_timeline.value())
        self.play_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.play_cap.read()
        if ok:
            self.play_frame_idx = idx + 1
            self.show_frame(frame)
            self.update_time_label()
        else:
            self.update_time_label()

    # --------- open / preview ----------
    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "", "Video files (*.mp4 *.avi *.mkv *.mov *.mpg *.mpeg);;All files (*)")
        if not path: return

        if self.src_cap: self.src_cap.release()
        self.src_path = path
        self.src_cap = cv2.VideoCapture(self.src_path)
        if not self.src_cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open video."); return

        fps = self.src_cap.get(cv2.CAP_PROP_FPS) or 0
        if fps > 0:
            try: self.spin_fps.setValue(float(fps))
            except: pass
            self.src_fps = float(self.spin_fps.value())
        else:
            self.src_fps = float(self.spin_fps.value())

        self.src_frame_count = int(self.src_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        ok, first = self.src_cap.read()
        if ok: self.show_frame(first)
        self.src_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.roi = None; self.label.clear_roi()
        self.invalidate_processed()
        self.statusBar().showMessage(f"Opened: {os.path.basename(self.src_path)} ({self.src_fps:.2f} fps, {self.src_frame_count} frames)")

    # --------- transform (offline) ----------
    def transform_video(self):
        if not self.src_cap or not self.src_path:
            QtWidgets.QMessageBox.warning(self, "No video", "Open a video first."); return

        x, y, w, h = (0, 0, int(self.src_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) \
                     if not self.roi else self.roi

        fps = float(self.spin_fps.value())
        low = float(self.spin_low.value())
        high = float(self.spin_high.value())
        high = min(high, fps/2 - 0.01)
        if high <= low: high = low + 0.01
        levels = int(self.spin_levels.value())

        # *** ključno: slider 1–100 → efektivno M = 10–1000
        alpha = float(self.slider_amp.value()) * 10.0

        base, _ = os.path.splitext(self.src_path)
        out_path = base + "_magnified.avi"   # MJPG for accurate seeking
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        cap = cv2.VideoCapture(self.src_path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to reopen source."); return
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot open VideoWriter."); return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        prog = QtWidgets.QProgressDialog("Transforming…", "Cancel", 0, total if total>0 else 0, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0)

        magnifier = None
        idx = 0
        ok, frame = cap.read()
        while ok:
            roi_view = frame[y:y+h, x:x+w]
            if roi_view.ndim == 2:
                gray = roi_view.astype(np.float32) / 255.0
            else:
                gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            if magnifier is None:
                magnifier = RieszMotionMagnifier(levels, low, high, fps, shape_hw=gray.shape)

            out = magnifier.magnify(gray, alpha)
            out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR)

            disp = frame.copy()
            disp[y:y+h, x:x+w] = out_bgr
            writer.write(disp)

            idx += 1
            if total > 0 and idx % 5 == 0:
                prog.setValue(idx)
                if prog.wasCanceled():
                    break

            ok, frame = cap.read()

        prog.setValue(total if total>0 else idx)
        cap.release(); writer.release()

        if prog.wasCanceled():
            try: os.remove(out_path)
            except: pass
            self.statusBar().showMessage("Transform canceled.")
            return

        if self.play_cap: self.play_cap.release()
        self.proc_path = out_path
        self.play_cap = cv2.VideoCapture(self.proc_path)
        self.play_frame_count = int(self.play_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.play_frame_idx = 0

        self.slider_timeline.setRange(0, max(0, self.play_frame_count - 1))
        ok, first_proc = self.play_cap.read()
        if ok:
            self.play_frame_idx = 1
            self.show_frame(first_proc)
        self.play_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.update_time_label()
        self.statusBar().showMessage(f"Transform done: {os.path.basename(self.proc_path)}")

    # --------- save processed (copy AVI) ----------
    def save_as(self):
        if not self.proc_path or not os.path.isfile(self.proc_path):
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Please run Transform first.")
            return
        dst, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save As", os.path.basename(self.proc_path), "AVI files (*.avi);;All files (*)")
        if not dst: return
        try:
            shutil.copyfile(self.proc_path, dst)
            self.statusBar().showMessage(f"Saved: {dst}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    # --------- export MP4 (H.264/MP4V fallback) ----------
    def export_mp4(self):
        if not self.proc_path or not os.path.isfile(self.proc_path):
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Please run Transform first.")
            return

        default_name = os.path.splitext(os.path.basename(self.proc_path))[0] + ".mp4"
        dst, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export MP4 (H.264/MP4)", default_name, "MP4 files (*.mp4)")
        if not dst:
            return

        cap = cv2.VideoCapture(self.proc_path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open processed video.")
            return

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.spin_fps.value())

        def try_writer(code):
            fourcc = cv2.VideoWriter_fourcc(*code)
            writer = cv2.VideoWriter(dst, fourcc, fps, (W, H))
            return writer if writer.isOpened() else None

        writer = None
        for code in ("avc1", "H264", "mp4v"):
            writer = try_writer(code)
            if writer is not None:
                chosen = code
                break

        if writer is None:
            QtWidgets.QMessageBox.critical(self, "Export failed", "No suitable MP4 encoder found (avc1/H264/mp4v).")
            cap.release()
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        prog = QtWidgets.QProgressDialog(f"Exporting MP4 ({chosen})…", "Cancel", 0, total if total>0 else 0, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0)

        idx = 0
        ok, frame = cap.read()
        while ok:
            writer.write(frame)
            idx += 1
            if total > 0 and idx % 5 == 0:
                prog.setValue(idx)
                if prog.wasCanceled():
                    break
            ok, frame = cap.read()

        prog.setValue(total if total>0 else idx)
        writer.release(); cap.release()

        if prog.wasCanceled():
            try: os.remove(dst)
            except: pass
            self.statusBar().showMessage("Export canceled.")
        else:
            self.statusBar().showMessage(f"Exported: {dst}")

    # --------- playback controls ----------
    def toggle_play(self, playing: bool):
        if not self.play_cap:
            QtWidgets.QMessageBox.information(self, "No processed video", "Run Transform first.")
            self.btn_play.setChecked(False)
            return
        if playing:
            fps = float(self.spin_fps.value())
            speed = max(1, int(self.slider_speed.value())) / 100.0  # 1%..100%
            interval_ms = max(1, int(1000 / max(fps * speed, 0.01)))
            self.timer.start(interval_ms)
            self.btn_play.setText("Pause")
        else:
            self.timer.stop()
            self.btn_play.setText("Play")

    def stop_playback(self):
        self.timer.stop()
        self.btn_play.setChecked(False)
        self.btn_play.setText("Play")
        if self.play_cap:
            self.play_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.play_cap.read()
            if ok:
                self.play_frame_idx = 1
                self.show_frame(frame)
            else:
                self.play_frame_idx = 0
            self.play_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            try:
                self.slider_timeline.blockSignals(True)
                self.slider_timeline.setValue(0)
            finally:
                self.slider_timeline.blockSignals(False)
            self.update_time_label()

    def next_frame_play(self):
        if not self.play_cap: return
        ok, frame = self.play_cap.read()
        if not ok:
            if self.chk_loop.isChecked():
                self.play_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.play_frame_idx = 0
                return
            else:
                self.btn_play.setChecked(False)
                return

        self.play_frame_idx += 1
        self.show_frame(frame)
        try:
            self.slider_timeline.blockSignals(True)
            self.slider_timeline.setValue(self.play_frame_idx - 1)
        finally:
            self.slider_timeline.blockSignals(False)
        self.update_time_label()

    # --------- ROI & UI helpers ----------
    def on_roi_changed(self, rect: QtCore.QRect):
        if rect is None:
            self.roi = None
        else:
            self.roi = (rect.x(), rect.y(), rect.width(), rect.height())

    def update_time_label(self):
        fps = float(self.spin_fps.value())
        cur = max(0, self.play_frame_idx - 1)
        t_cur = self.format_time(cur, fps)
        t_tot = self.format_time(self.play_frame_count, fps) if self.play_frame_count > 0 else "00:00"
        self.lbl_time.setText(f"{t_cur} / {t_tot}")

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

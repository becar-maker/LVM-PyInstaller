from PySide6 import QtCore, QtGui, QtWidgets

class VideoLabel(QtWidgets.QLabel):
    roiChanged = QtCore.Signal(object)  # QRect or None

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color:#111; color:#ccc;")
        self._roi=None; self._dragging=False; self._start=None
        self.setMouseTracking(True)

    def clear_roi(self):
        self._roi=None; self.update(); self.roiChanged.emit(None)

    def mousePressEvent(self, e:QtGui.QMouseEvent):
        if e.button()==QtCore.Qt.LeftButton and self.pixmap() is not None:
            self._dragging=True; self._start=e.position().toPoint()

    def mouseMoveEvent(self, e:QtGui.QMouseEvent):
        if self._dragging and self._start is not None:
            end=e.position().toPoint()
            r=QtCore.QRect(self._start,end).normalized().intersected(self.rect())
            self._roi=r; self.update()

    def mouseReleaseEvent(self, e:QtGui.QMouseEvent):
        if e.button()==QtCore.Qt.LeftButton and self._dragging:
            self._dragging=False
            if self._roi and self._roi.width()>5 and self._roi.height()>5:
                self.roiChanged.emit(self._roi)
            else: self.clear_roi()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._roi:
            p=QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setPen(QtGui.QPen(QtGui.QColor(0,200,255),2,QtCore.Qt.DashLine))
            p.drawRect(self._roi)


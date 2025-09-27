import numpy as np
from .riesz import RieszMotionMagnifier
from .backend_interface import MagnifierBackend

class RieszBackend(MagnifierBackend):
    def __init__(self):
        self._impl = None
        self._cfg = None

    def reset(self, levels, low, high, fps, shape_hw):
        self._cfg = (levels, low, high, fps, shape_hw)
        self._impl = RieszMotionMagnifier(levels, low, high, fps, shape_hw)

    def magnify(self, gray01, alpha):
        if self._impl is None or self._cfg is None:
            raise RuntimeError("RieszBackend not initialized (call reset).")
        out = self._impl.magnify(gray01, float(alpha))
        return np.clip(out, 0.0, 1.0).astype(np.float32)


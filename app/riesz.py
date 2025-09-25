import numpy as np
import cv2
import math

_EPS = 1e-8

def _pyr_down(img):
    return cv2.pyrDown(img)

def _pyr_up(img, dst_size):
    up = cv2.pyrUp(img)
    if up.shape[:2] != dst_size:
        up = cv2.resize(up, (dst_size[1], dst_size[0]), interpolation=cv2.INTER_LINEAR)
    return up

def _build_laplacian_pyramid(img: np.ndarray, levels: int):
    """Return (lap_levels:list[H×W], residual: H/2^L × W/2^L)."""
    g = [img]
    for _ in range(levels):
        g.append(_pyr_down(g[-1]))
    lap = []
    for i in range(levels):
        up = _pyr_up(g[i+1], g[i].shape[:2])
        lap.append(g[i] - up)
    return lap, g[-1]

def _reconstruct_from_laplacian(lap, residual):
    img = residual
    for i in reversed(range(len(lap))):
        img = _pyr_up(img, lap[i].shape[:2]) + lap[i]
    return img

def _riesz_transform(band: np.ndarray):
    """
    Aproksimacija 2D Riesz transformacije prek gradientov (Sobel).
    Vrne (Rx, Ry), Rnorm in monogenic amplitude A = sqrt(band^2 + Rnorm^2)
    ter fazo phi = atan2(Rnorm, band).
    """
    Rx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Ry = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    Rnorm = np.sqrt(Rx*Rx + Ry*Ry) + _EPS
    A = np.sqrt(band*band + Rnorm*Rnorm) + _EPS
    phi = np.arctan2(Rnorm, band + _EPS)  # monogenic phase
    return Rx, Ry, Rnorm, A, phi

class _IIRBandpass:
    """Dvo-polni IIR band-pass za fazo (na vsakem piksli posebej)."""
    def __init__(self, low_hz: float, high_hz: float, fps: float, shape):
        aH = math.exp(-2*math.pi*high_hz / fps)
        aL = math.exp(-2*math.pi*low_hz  / fps)
        self.aH = np.float32(aH)
        self.aL = np.float32(aL)
        self.yH = np.zeros(shape, np.float32)
        self.yL = np.zeros(shape, np.float32)
        self.initialized = False

    def update(self, x: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.yH[...] = x; self.yL[...] = x
            self.initialized = True
        self.yH = (1.0 - self.aH) * x + self.aH * self.yH
        self.yL = (1.0 - self.aL) * x + self.aL * self.yL
        return self.yH - self.yL

class RieszMotionMagnifier:
    """
    Phase-based motion magnification z monogenic (Riesz) signalom po Laplaceovi piramidi.
    - levels: št. nivojev piramide (1..5)
    - low/high: časovni pas v Hz
    - fps: vzorčna frekvenca
    """
    def __init__(self, levels: int, low_hz: float, high_hz: float, fps: float, shape_hw):
        self.levels = int(max(1, min(5, levels)))
        self.low = float(low_hz)
        self.high = float(high_hz)
        self.fps = float(fps)

        # IIR za fazo na vsakem nivoju
        self.filters = None
        self._init_filters(shape_hw)

    def _init_filters(self, shape_hw):
        # zgradi dummy piramido samo za oblike nivojev
        zeros = np.zeros(shape_hw, np.float32)
        lap, _ = _build_laplacian_pyramid(zeros, self.levels)
        self.filters = [ _IIRBandpass(self.low, self.high, self.fps, l.shape) for l in lap ]

    def reinit(self, levels: int, low_hz: float, high_hz: float, fps: float, shape_hw):
        self.levels = int(max(1, min(5, levels)))
        self.low = float(low_hz)
        self.high = float(high_hz)
        self.fps = float(fps)
        self._init_filters(shape_hw)

    def magnify(self, gray01: np.ndarray, alpha: float) -> np.ndarray:
        """
        Vhod: gray01 (float32, 0..1), izhod: magnified gray v 0..1 (float32).
        """
        img = gray01.astype(np.float32)

        # 1) Laplaceova piramida
        lap, residual = _build_laplacian_pyramid(img, self.levels)

        # 2) Po nivojih: monogenic faza -> IIR band-pass -> amplifikacija -> rekonstrukcija banda
        out_lap = []
        for i, band in enumerate(lap):
            Rx, Ry, Rnorm, A, phi = _riesz_transform(band)
            bp = self.filters[i].update(phi)  # band-passed phase
            phi_amp = phi + alpha * bp
            # rekonstrukcija banda iz amplitude in nove faze
            band_out = A * np.cos(phi_amp)
            out_lap.append(band_out)

        # 3) Rekonstrukcija slike
        out = _reconstruct_from_laplacian(out_lap, residual)

        # 4) Omeji na 0..1
        return np.clip(out, 0.0, 1.0)


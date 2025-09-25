import numpy as np
import cv2
import math

_EPS = 1e-8

def _pyr_down(img):
    return cv2.pyrDown(img)

def _pyr_up(img, dst_size_hw):
    up = cv2.pyrUp(img)
    if up.shape[:2] != dst_size_hw:
        # cv2.resize: (width, height) = (cols, rows)
        up = cv2.resize(up, (dst_size_hw[1], dst_size_hw[0]), interpolation=cv2.INTER_LINEAR)
    return up

def _build_laplacian_pyramid(img: np.ndarray, levels: int):
    """
    Vrne (seznam Laplace bandov [lvl0..], residual na dnu).
    lvl0 = najvišja ločljivost (najfinejši band).
    """
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

def _riesz_transform_sobel(band: np.ndarray):
    """
    Hitra aproksimacija 2D Riesz z gradientoma (Sobel).
    Vrne Rx, Ry, njihovo normo R, monogenic amplitudo A in fazo phi.
    """
    Rx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Ry = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    R  = np.sqrt(Rx*Rx + Ry*Ry) + _EPS
    A  = np.sqrt(band*band + R*R) + _EPS
    phi = np.arctan2(R, band + _EPS)  # približna “monogenic” faza (0..π)
    return Rx, Ry, R, A, phi

class _IIRBandpass:
    """Preprost dvo-polni IIR band-pass (po pikslih) za fazo."""
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
    Phase-based motion magnification (Sobel-Riesz + Laplace piramida + časovni band-pass).
    `alpha` tukaj pomeni **neposreden faktor premika M** (npr. 5, 50, 10000):
        u' = M * u   → realizirano kot fazna ojačitev: phi' = phi + (M-1) * Δphi_bp
    Opomba: ker je to hitra aproksimacija brez unwrapa, je zelo linearna za zmerne M
    in robustna po hitrosti. Pri zelo velikih M na finih detajlih pričakuj artefakte.
    """
    def __init__(self, levels: int, low_hz: float, high_hz: float, fps: float, shape_hw):
        self.levels = int(max(1, min(5, levels)))
        self.low = float(low_hz)
        self.high = float(high_hz)
        self.fps = float(fps)
        self._init_filters(shape_hw)

    def _init_filters(self, shape_hw):
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
        Vhod: gray01 (float32, 0..1); Izhod: magnified gray (float32, 0..1).
        `alpha` = M (ciljni faktor premika). Ojačamo le band-pass del faze.
        """
        img = gray01.astype(np.float32)

        # 1) Laplaceova piramida
        lap, residual = _build_laplacian_pyramid(img, self.levels)

        # 2) Po nivojih: Riesz (Sobel) → faza → IIR band-pass → ojačanje → rekonstrukcija
        out_lap = []
        phase_gain = (alpha - 1.0)  # npr. 50 → Δφ * 49
        for i, band in enumerate(lap):
            Rx, Ry, R, A, phi = _riesz_transform_sobel(band)
            dphi_bp = self.filters[i].update(phi)  # band-pass faza

            # uniformno povečaj band-pass del faze
            phi_amp = phi + phase_gain * dphi_bp

            # rekonstrukcija even-komponente (Laplacian band) iz amplitude in faze
            band_out = A * np.cos(phi_amp)
            out_lap.append(band_out)

        # 3) Rekonstrukcija slike in omejitev
        out = _reconstruct_from_laplacian(out_lap, residual)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

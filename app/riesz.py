import numpy as np
import cv2
import math

_EPS = 1e-8
_TWO_PI = 2.0 * math.pi

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
    Vrne (seznam Laplace bandov [lvl0..lvlL-1], residual na dnu).
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

def _riesz_fft(band: np.ndarray):
    """
    Točna 2D Riesz transformacija v frekvenčni domeni.
    Vrne (R_x, R_y), njuna norma R = sqrt(R_x^2 + R_y^2),
    monogenic amplitudo A = sqrt(band^2 + R^2) in fazo phi = atan2(R, band).
    """
    H, W = band.shape
    # frekvenčne mreže (radiani na piksel)
    ky = np.fft.fftfreq(H) * _TWO_PI  # po vrsticah (y)
    kx = np.fft.fftfreq(W) * _TWO_PI  # po stolpcih (x)
    KX, KY = np.meshgrid(kx, ky)       # oblike (H, W)
    OMEGA = np.sqrt(KX*KX + KY*KY)
    # Riesz filtri v ω-dom.
    Hx = 1j * np.divide(KX, OMEGA, out=np.zeros_like(KX, dtype=np.complex64), where=OMEGA>0)
    Hy = 1j * np.divide(KY, OMEGA, out=np.zeros_like(KY, dtype=np.complex64), where=OMEGA>0)

    F = np.fft.fft2(band)
    Rx = np.fft.ifft2(Hx * F).real.astype(np.float32)
    Ry = np.fft.ifft2(Hy * F).real.astype(np.float32)
    R = np.sqrt(Rx*Rx + Ry*Ry) + _EPS
    A = np.sqrt(band*band + R*R) + _EPS
    phi = np.arctan2(R, band + _EPS)  # [0, π]
    return Rx, Ry, R, A, phi

class _IIRBandpass:
    """Dvo-polni IIR band-pass (po pikslih) za fazo (že unwrapano)."""
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

def _unwrap_step(phi_curr: np.ndarray, phi_prev: np.ndarray) -> np.ndarray:
    """
    Enokaderna časovna razvezava faze:
    prilagodi phi_curr tako, da je (phi_curr - phi_prev) v (-π, π].
    Vrne unwrapano trenutno fazo.
    """
    d = phi_curr - phi_prev
    d_wrapped = (d + math.pi) % _TWO_PI - math.pi
    return phi_prev + d_wrapped

class RieszMotionMagnifier:
    """
    Phase-based motion magnification (monogenic/Riesz + Laplaceova piramida + časovni band-pass).

    ***Pomembno***: parameter `alpha` v `magnify(gray01, alpha)` je neposreden
    **faktor premika M** za band-pass del gibanja:
        u' = M * u   (M = alpha; npr. 5.0, 50.0)
    Implementirano je:
      - točna Riesz transformacija (FFT),
      - časovna razvezava (unwrap) faze na vsaki ravni,
      - IIR band-pass filt. faze,
      - fazna ojačitev: phi' = phi + (M-1) * Δphi_bp,
      - rekonstrukcija even-komponente iz A * cos(phi').
    """
    def __init__(self, levels: int, low_hz: float, high_hz: float, fps: float, shape_hw):
        self.levels = int(max(1, min(5, levels)))
        self.low = float(low_hz)
        self.high = float(high_hz)
        self.fps = float(fps)

        self.filters = None       # IIR bandpass per level
        self.prev_phi = None      # unwrap state per level
        self._init_state(shape_hw)

    def _init_state(self, shape_hw):
        zeros = np.zeros(shape_hw, np.float32)
        lap, _ = _build_laplacian_pyramid(zeros, self.levels)
        self.filters = [ _IIRBandpass(self.low, self.high, self.fps, l.shape) for l in lap ]
        self.prev_phi = [ None for _ in lap ]

    def reinit(self, levels: int, low_hz: float, high_hz: float, fps: float, shape_hw):
        self.levels = int(max(1, min(5, levels)))
        self.low = float(low_hz)
        self.high = float(high_hz)
        self.fps = float(fps)
        self._init_state(shape_hw)

    def magnify(self, gray01: np.ndarray, alpha: float) -> np.ndarray:
        """
        Vhod: gray01 (float32, 0..1)
        Izhod: magnified gray (float32, 0..1)

        `alpha` = M (ciljni faktor premika).
        Delujemo okvir-po-okvir; za linearno obnašanje je pomembno, da kličete
        magnify() na istih zaporednih sličicah (kar vaša Transform pot zagotavlja).
        """
        img = gray01.astype(np.float32)

        # 1) Laplaceova piramida trenutnega kadra
        lap, residual = _build_laplacian_pyramid(img, self.levels)

        out_lap = []
        phase_gain = (alpha - 1.0)  # npr. 50 → Δφ * 49
        for i, band in enumerate(lap):
            # 2) Točna Riesz + monogenic A/phi
            Rx, Ry, R, A, phi = _riesz_fft(band)

            # 3) Časovna razvezava faze (glede na prejšnji kader te ravni)
            if self.prev_phi[i] is None:
                phi_unw = phi.copy()
            else:
                phi_unw = _unwrap_step(phi, self.prev_phi[i])
            self.prev_phi[i] = phi_unw

            # 4) Band-pass na unwrapani fazi
            dphi_bp = self.filters[i].update(phi_unw)  # Δφ_bp (v radianih)

            # 5) Ojačanje fazne spremembe tako, da u' = alpha * u
            phi_amp = phi_unw + phase_gain * dphi_bp

            # 6) Rekonstrukcija even-komponente
            band_out = A * np.cos(phi_amp)
            out_lap.append(band_out)

        # 7) Rekonstrukcija slike in omejitev
        out = _reconstruct_from_laplacian(out_lap, residual)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

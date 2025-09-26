import numpy as np
import cv2
import math

_EPS = 1e-8

# ---------- Laplaceova piramida ----------
def _pyr_down(img):
    return cv2.pyrDown(img)

def _pyr_up(img, dst_size_hw):
    up = cv2.pyrUp(img)
    if up.shape[:2] != dst_size_hw:
        up = cv2.resize(up, (dst_size_hw[1], dst_size_hw[0]), interpolation=cv2.INTER_LINEAR)
    return up

def _build_laplacian_pyramid(img: np.ndarray, levels: int):
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

# ---------- Riesz 3×3 jedra (kot v MATLAB varianti) ----------
KX = np.array([[0, 0, 0],
               [0.5, 0, -0.5],
               [0, 0, 0]], np.float32)
KY = np.array([[0, 0.5, 0],
               [0, 0, 0],
               [0, -0.5, 0]], np.float32)

def _riesz_components(band: np.ndarray):
    Rx = cv2.filter2D(band, cv2.CV_32F, KX, borderType=cv2.BORDER_REPLICATE)
    Ry = cv2.filter2D(band, cv2.CV_32F, KY, borderType=cv2.BORDER_REPLICATE)
    return Rx, Ry

# ---------- preprost IIR band-pass (razlika dveh enopolnih LP) ----------
class _IIRBandpass:
    """
    Dva enopolna LP filtra pri fH in fL → band = LP(fH) - LP(fL).
    Stabilno, hitro, brez dodatne normalizacije.
    """
    def __init__(self, low_hz: float, high_hz: float, fps: float, shape):
        self.aH = float(math.exp(-2 * math.pi * high_hz / fps))
        self.aL = float(math.exp(-2 * math.pi * low_hz  / fps))
        self.yH = np.zeros(shape, np.float32)
        self.yL = np.zeros(shape, np.float32)
        self.initialized = False

    def update(self, x: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.yH[...] = x
            self.yL[...] = x
            self.initialized = True
        self.yH = (1.0 - self.aH) * x + self.aH * self.yH
        self.yL = (1.0 - self.aL) * x + self.aL * self.yL
        return self.yH - self.yL

# ---------- amplitudno uteženo glajenje ----------
def _amp_weighted_blur(signal: np.ndarray, amplitude: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return signal
    ksz = int(4 * sigma + 1) | 1
    w = amplitude + 1e-8
    num = cv2.GaussianBlur(signal * w, (ksz, ksz), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(w,             (ksz, ksz), sigma, borderType=cv2.BORDER_REFLECT)
    return num / (den + 1e-8)

# ---------- glavna klasa ----------
class RieszMotionMagnifier:
    """
    MATLAB-like kvaternionična Riesz magnifikacija gibanja.
    `alpha` = M (Amplification: 1..100) — ciljni faktor premika za band-pass komponento.
    Koraki:
      - Laplace piramida
      - Riesz (3×3) → q = (a, Rx, Ry)
      - kvaternionična fazna razlika: q_cur * conj(q_prev) → (r, vx, vy)
      - časovna kumulacija (unwrap) projekcij (cos, sin)
      - IIR band-pass (brez normalizacije)
      - soft maska m = A/(A+τ) in amplitudno uteženo glajenje (σ)
      - ojačanje (M-1), phase-shift, vrnemo nov even del (band_out)
    """
    def __init__(self, levels: int, low_hz: float, high_hz: float, fps: float, shape_hw):
        self.levels = int(max(1, min(5, levels)))
        self.low = float(low_hz)
        self.high = float(high_hz)
        self.fps = float(fps)

        # Nastavitve stabilnosti
        self.blur_sigma = 1.2     # Gauss σ za amplitudno uteženo glajenje (0.0 za izklop)
        self.tau_mask   = 0.2     # τ v m = A/(A+τ); večji τ → manj ojačanja v šibkih območjih

        # per-level stanje
        self.prev_a = None
        self.prev_rx = None
        self.prev_ry = None
        self.cum_cos = None
        self.cum_sin = None
        self.bp_cos = None
        self.bp_sin = None
        self.level_gain = None  # per-skalen gain (privzeto 1.0)
        self._init_state(shape_hw)

    def _init_state(self, shape_hw):
        zeros = np.zeros(shape_hw, np.float32)
        lap, _ = _build_laplacian_pyramid(zeros, self.levels)
        shp = [l.shape for l in lap]

        self.prev_a  = [ None for _ in shp ]
        self.prev_rx = [ None for _ in shp ]
        self.prev_ry = [ None for _ in shp ]

        self.cum_cos = [ np.zeros(s, np.float32) for s in shp ]
        self.cum_sin = [ np.zeros(s, np.float32) for s in shp ]

        self.bp_cos  = [ _IIRBandpass(self.low, self.high, self.fps, s) for s in shp ]
        self.bp_sin  = [ _IIRBandpass(self.low, self.high, self.fps, s) for s in shp ]

        self.level_gain = [1.0 for _ in shp]  # po potrebi lahko kasneje uglasimo

    def reinit(self, levels: int, low_hz: float, high_hz: float, fps: float, shape_hw):
        self.levels = int(max(1, min(5, levels)))
        self.low = float(low_hz)
        self.high = float(high_hz)
        self.fps = float(fps)
        self._init_state(shape_hw)

    def magnify(self, gray01: np.ndarray, alpha: float) -> np.ndarray:
        """
        Vhod: gray01 (float32, 0..1) → Izhod: magnified gray (float32, 0..1)
        """
        alpha = float(np.clip(alpha, 1.0, 100.0))  # varnostna omejitev

        img = gray01.astype(np.float32)

        # 1) Laplaceova piramida
        lap, residual = _build_laplacian_pyramid(img, self.levels)

        out_lap = []
        gain = alpha - 1.0
        for i, band in enumerate(lap):
            a = band.astype(np.float32)  # even
            Rx, Ry = _riesz_components(a)  # odd
            A = np.sqrt(a*a + Rx*Rx + Ry*Ry) + _EPS  # amplituda koeficienta

            pa  = self.prev_a[i]
            prx = self.prev_rx[i]
            pry = self.prev_ry[i]

            if pa is None:
                self.prev_a[i]  = a.copy()
                self.prev_rx[i] = Rx.copy()
                self.prev_ry[i] = Ry.copy()
                out_lap.append(a)
                continue

            # 2) kvaternionična fazna razlika: q_cur * conj(q_prev) → (r, vx, vy)
            r  = a*pa + Rx*prx + Ry*pry
            vx = pa*Rx - a*prx
            vy = pa*Ry - a*pry

            norm = np.sqrt(r*r + vx*vx + vy*vy) + _EPS
            # fazni kot v [0, π]
            phase = np.clip(r / norm, -1.0, 1.0)
            phase = np.arccos(phase)

            vnorm = np.sqrt(vx*vx + vy*vy) + _EPS
            cos_t = vx / vnorm
            sin_t = vy / vnorm

            # 3) časovna kumulacija (unwrap)
            self.cum_cos[i] += phase * cos_t
            self.cum_sin[i] += phase * sin_t

            # 4) IIR band-pass (brez normalizacije)
            fcos = self.bp_cos[i].update(self.cum_cos[i])
            fsin = self.bp_sin[i].update(self.cum_sin[i])

            # 5) soft maska po amplitudi (stabilnost v šibkih območjih)
            m = A / (A + self.tau_mask)
            fcos *= m
            fsin *= m

            # 6) amplitudno uteženo glajenje
            if self.blur_sigma > 0.0:
                fcos = _amp_weighted_blur(fcos, A, self.blur_sigma)
                fsin = _amp_weighted_blur(fsin, A, self.blur_sigma)

            # 7) per-skalen gain (privzeto 1.0; možnost uglasitve)
            g = float(self.level_gain[i])
            fcos *= g
            fsin *= g

            # 8) ojačanje (M-1)
            fcos *= gain
            fsin *= gain

            # 9) phase-shift: q_out = exp(n*θ) * q_cur
            mag = np.sqrt(fcos*fcos + fsin*fsin) + _EPS
            exp_r  = np.cos(mag)
            s_over = np.sin(mag) / mag
            ex = fcos * s_over
            ey = fsin * s_over

            a_out = exp_r * a - ex * Rx - ey * Ry
            out_lap.append(a_out)

            # update prev
            self.prev_a[i][...]  = a
            self.prev_rx[i][...] = Rx
            self.prev_ry[i][...] = Ry

        out = _reconstruct_from_laplacian(out_lap, residual)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

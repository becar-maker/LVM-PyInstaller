import numpy as np
import cv2
import math

_EPS = 1e-8
PI = math.pi

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

# ---------- normaliziran IIR band-pass ----------
def _lp_mag(a: float, omega: float) -> float:
    """
    Magnituda enopolnega LP filtra y = (1-a)x + a y[-1] pri kotni frekvenci omega (rad/sample).
    |H_lp(e^jω)| = (1-a) / sqrt(1 + a^2 - 2 a cos ω)
    """
    num = 1.0 - a
    den = math.sqrt(1.0 + a*a - 2.0*a*math.cos(omega))
    return num / max(den, 1e-12)

class _NormIIRBandpass:
    """
    Band-pass kot razlika dveh enopolnih LP (pri fH in fL), a z **normalizacijo gaina**
    na središčni frekvenci fC = sqrt(fL * fH). To stabilizira amplitudo Δφ, da se
    ojačanje M obnaša linearno (npr. 30× → 60× ≈ 2× večji premik).
    """
    def __init__(self, low_hz: float, high_hz: float, fps: float, shape):
        assert low_hz > 0 and high_hz > low_hz and fps > 0
        self.aH = float(math.exp(-2 * math.pi * high_hz / fps))
        self.aL = float(math.exp(-2 * math.pi * low_hz  / fps))
        self.yH = np.zeros(shape, np.float32)
        self.yL = np.zeros(shape, np.float32)
        self.initialized = False

        # center frequency & normalizacija magnitude pri fC
        fC = math.sqrt(low_hz * high_hz)
        omega_c = 2.0 * math.pi * (fC / fps)
        gH = _lp_mag(self.aH, omega_c)
        gL = _lp_mag(self.aL, omega_c)
        G = gH - gL
        # če je G zelo majhen (robni primer), se izognemo eksploziji gaina
        self.norm = 1.0 / G if abs(G) > 1e-6 else 1.0

    def update(self, x: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.yH[...] = x
            self.yL[...] = x
            self.initialized = True
        # LP s poloma aH in aL
        self.yH = (1.0 - self.aH) * x + self.aH * self.yH
        self.yL = (1.0 - self.aL) * x + self.aL * self.yL
        # normaliziran band-pass
        return (self.yH - self.yL) * self.norm

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
    MATLAB-like kvaternionična Riesz magnifikacija z **normaliziranim band-passom**.
    `alpha` = M (Amplification: 1..100) — ciljni faktor premika band-pass komponente.
    """
    def __init__(self, levels: int, low_hz: float, high_hz: float, fps: float, shape_hw):
        self.levels = int(max(1, min(5, levels)))
        self.low = float(low_hz)
        self.high = float(high_hz)
        self.fps = float(fps)

        # manj glajenja, da se razlike M bolje vidijo; po potrebi zvišaj na 1.0
        self.blur_sigma = 0.5

        # per-level stanje
        self.prev_a = None
        self.prev_rx = None
        self.prev_ry = None
        self.cum_cos = None
        self.cum_sin = None
        self.bp_cos = None
        self.bp_sin = None
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

        self.bp_cos  = [ _NormIIRBandpass(self.low, self.high, self.fps, s) for s in shp ]
        self.bp_sin  = [ _NormIIRBandpass(self.low, self.high, self.fps, s) for s in shp ]

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

            # q_cur * conj(q_prev) → (r, vx, vy)
            r  = a*pa + Rx*prx + Ry*pry
            vx = pa*Rx - a*prx
            vy = pa*Ry - a*pry

            norm = np.sqrt(r*r + vx*vx + vy*vy) + _EPS
            phase = np.clip(r / norm, -1.0, 1.0)
            phase = np.arccos(phase)

            vnorm = np.sqrt(vx*vx + vy*vy) + _EPS
            cos_t = vx / vnorm
            sin_t = vy / vnorm

            # časovna kumulacija projekcij (unwrap)
            self.cum_cos[i] += phase * cos_t
            self.cum_sin[i] += phase * sin_t

            # **normaliziran** IIR band-pass
            fcos = self.bp_cos[i].update(self.cum_cos[i])
            fsin = self.bp_sin[i].update(self.cum_sin[i])

            # amplitudno uteženo glajenje (manjše kot prej)
            if self.blur_sigma > 0.0:
                fcos = _amp_weighted_blur(fcos, A, self.blur_sigma)
                fsin = _amp_weighted_blur(fsin, A, self.blur_sigma)

            # ojačanje (M-1)
            gain = (alpha - 1.0)
            fcos *= gain
            fsin *= gain

            # phase-shift: q_out = exp(n*θ) * q_cur
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

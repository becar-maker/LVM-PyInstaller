import numpy as np
import cv2
import math

_EPS = 1e-8

# ==============================
#   Laplacian piramida (Burt–Adelson, symmetric robovi)
# ==============================
# 1D binomsko jedro [1 4 6 4 1]/16
_G1D = np.array([1, 4, 6, 4, 1], np.float32) / 16.0
_BORDER = cv2.BORDER_REFLECT_101  # "symmetric" kot v MATLAB-u

def _sep_gauss5(img: np.ndarray) -> np.ndarray:
    """Separable 5-tap filter z 'symmetric' robovi."""
    return cv2.sepFilter2D(img, ddepth=-1, kernelX=_G1D, kernelY=_G1D, borderType=_BORDER)

def _reduce(img: np.ndarray) -> np.ndarray:
    """G_{i+1} = downsample2( gaussian5 * G_i )"""
    low = _sep_gauss5(img)
    return low[::2, ::2].copy()

def _expand(img: np.ndarray, dst_hw) -> np.ndarray:
    """
    Expand G_{i+1} nazaj na velikost G_i:
      - vstavimo ničle med piksle
      - filtriramo z 4x gauss5 (klasično expand za ohranitev energije)
    """
    H, W = dst_hw
    up = np.zeros((H, W), np.float32)
    up[::2, ::2] = img
    # 4x faktor po Burt–Adelsonu
    k = _G1D * 4.0
    up = cv2.sepFilter2D(up, ddepth=-1, kernelX=k, kernelY=k, borderType=_BORDER)
    return up

def _build_laplacian_pyramid(img: np.ndarray, levels: int):
    """
    Vrne (liste Laplace bandov [lvl0..lvlL-1], residual na dnu).
    lvl0 je najvišja ločljivost (najfinejši band).
    """
    g = [img]
    for _ in range(levels):
        g.append(_reduce(g[-1]))
    lap = []
    for i in range(levels):
        up = _expand(g[i+1], g[i].shape[:2])
        # poravnava dimezije (če pride do +-1 zaradi deljenja)
        if up.shape != g[i].shape:
            up = cv2.resize(up, (g[i].shape[1], g[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        lap.append(g[i] - up)
    return lap, g[-1]

def _reconstruct_from_laplacian(lap, residual):
    img = residual
    for i in reversed(range(len(lap))):
        up = _expand(img, lap[i].shape[:2])
        if up.shape != lap[i].shape:
            up = cv2.resize(up, (lap[i].shape[1], lap[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        img = up + lap[i]
    return img

# ==============================
#   Riesz 3×3 jedra (kot v MATLAB)
# ==============================
KX = np.array([[0, 0, 0],
               [0.5, 0, -0.5],
               [0, 0, 0]], np.float32)
KY = np.array([[0, 0.5, 0],
               [0, 0, 0],
               [0, -0.5, 0]], np.float32)

def _riesz_components(band: np.ndarray):
    Rx = cv2.filter2D(band, cv2.CV_32F, KX, borderType=_BORDER)
    Ry = cv2.filter2D(band, cv2.CV_32F, KY, borderType=_BORDER)
    return Rx, Ry

# ==============================
#   IIR band-pass (difference-of-IIR, kot v .m)
# ==============================
class _IIRBandpass:
    """
    Dva enopolna LP filtra pri fH in fL → band = LP(fH) - LP(fL).
    Inicializacija na x (kot v praksi v MATLAB skriptih) za hiter "warm-up".
    """
    def __init__(self, low_hz: float, high_hz: float, fps: float, shape):
        assert high_hz > low_hz > 0 and fps > 0
        self.aH = float(math.exp(-2.0 * math.pi * high_hz / fps))
        self.aL = float(math.exp(-2.0 * math.pi * low_hz  / fps))
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

# ==============================
#   AmplitudeWeightedBlur + soft mask
# ==============================
def _amp_weighted_blur(signal: np.ndarray, amplitude: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return signal
    ksz = int(4 * sigma + 1) | 1
    w = amplitude + 1e-8
    num = cv2.GaussianBlur(signal * w, (ksz, ksz), sigma, borderType=_BORDER)
    den = cv2.GaussianBlur(w,             (ksz, ksz), sigma, borderType=_BORDER)
    return num / (den + 1e-8)

# ==============================
#   Glavna klasa
# ==============================
class RieszMotionMagnifier:
    """
    MATLAB-like Riesz motion magnification:
      - Laplacian piramida (Burt–Adelson; symmetric robovi)
      - Riesz 3×3 jedra → q = (a, Rx, Ry)
      - kvaternionična fazna razlika: q_cur * conj(q_prev) → (r, vx, vy)
      - časovna kumulacija (unwrap) projekcij (cos, sin)
      - IIR band-pass (difference-of-IIR)
      - soft maska m = A/(A+τ) + AmplitudeWeightedBlur(σ)
      - phase-shift: q_out = exp(n*θ) * q_cur → vzamemo real(q_out) kot band_out
    `alpha` = M (Amplification: 1..100) — ciljni faktor premika band-pass komponent.
    """
    def __init__(self, levels: int, low_hz: float, high_hz: float, fps: float, shape_hw):
        self.levels = int(max(1, min(5, levels)))
        self.low = float(low_hz)
        self.high = float(high_hz)
        self.fps = float(fps)

        # Parametri stabilnosti (po vzoru MATLAB utilityjev)
        self.blur_sigma = 1.0  # Gauss σ za AW blur (0.0 za izklop)
        self.tau_mask   = 0.2  # τ v m = A/(A+τ); večji τ → manj ojačanja v šibkih območjih

        # Stanje per-level
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

        self.bp_cos  = [ _IIRBandpass(self.low, self.high, self.fps, s) for s in shp ]
        self.bp_sin  = [ _IIRBandpass(self.low, self.high, self.fps, s) for s in shp ]

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
        gain = alpha - 1.0

        img = gray01.astype(np.float32)

        # 1) Laplacian piramida
        lap, residual = _build_laplacian_pyramid(img, self.levels)

        out_lap = []
        for i, band in enumerate(lap):
            a = band.astype(np.float32)  # even komponenta
            Rx, Ry = _riesz_components(a)  # odd komponenti
            A = np.sqrt(a*a + Rx*Rx + Ry*Ry) + _EPS  # amplituda koeficienta (za masko/glajenje)

            pa  = self.prev_a[i]
            prx = self.prev_rx[i]
            pry = self.prev_ry[i]

            if pa is None:
                # init: zapomni si trenutne koeficiente; brez faznega zamika
                self.prev_a[i]  = a.copy()
                self.prev_rx[i] = Rx.copy()
                self.prev_ry[i] = Ry.copy()
                out_lap.append(a)
                continue

            # 2) kvaternionična fazna razlika med zaporednima koeficientoma
            # q_cur * conj(q_prev) → (r, vx, vy); (z-komponenta tukaj ni uporabljena)
            r  = a*pa + Rx*prx + Ry*pry
            vx = pa*Rx - a*prx
            vy = pa*Ry - a*pry

            norm = np.sqrt(r*r + vx*vx + vy*vy) + _EPS
            # fazni kot [0..π]
            phase = np.clip(r / norm, -1.0, 1.0)
            phase = np.arccos(phase)

            vnorm = np.sqrt(vx*vx + vy*vy) + _EPS
            cos_t = vx / vnorm
            sin_t = vy / vnorm

            # 3) časovna kumulacija (unwrap) projekcij
            self.cum_cos[i] += phase * cos_t
            self.cum_sin[i] += phase * sin_t

            # 4) IIR band-pass na kumuliranih komponentah (brez normalizacij)
            fcos = self.bp_cos[i].update(self.cum_cos[i])
            fsin = self.bp_sin[i].update(self.cum_sin[i])

            # 5) soft maska + AW blur
            m = A / (A + self.tau_mask)
            fcos *= m
            fsin *= m

            if self.blur_sigma > 0.0:
                fcos = _amp_weighted_blur(fcos, A, self.blur_sigma)
                fsin = _amp_weighted_blur(fsin, A, self.blur_sigma)

            # 6) ojačanje (M-1)
            fcos *= gain
            fsin *= gain

            # 7) phase-shift: q_out = exp(n*θ) * q_cur
            mag = np.sqrt(fcos*fcos + fsin*fsin) + _EPS
            exp_r  = np.cos(mag)
            s_over = np.sin(mag) / mag
            ex = fcos * s_over
            ey = fsin * s_over

            # real(q_out) = exp_r * a - ex * Rx - ey * Ry
            a_out = exp_r * a - ex * Rx - ey * Ry
            out_lap.append(a_out)

            # 8) posodobitev prejšnjih koeficientov
            self.prev_a[i][...]  = a
            self.prev_rx[i][...] = Rx
            self.prev_ry[i][...] = Ry

        # 9) rekonstrukcija slike
        out = _reconstruct_from_laplacian(out_lap, residual)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

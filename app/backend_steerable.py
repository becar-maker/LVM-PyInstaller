import numpy as np
from .backend_interface import MagnifierBackend

class SteerableBackend(MagnifierBackend):
    """
    Phase-based magnifier prek kompleksne steerable piramide (pyrtools).
    - Decompose: complex steerable pyramid (frequency domain)
    - Temporal IIR band-pass na fazi (vsak band/orientacija posebej)
    - Phase push: phi' = phi + alpha * BPF(phi)
    - Reconstruct
    """
    def __init__(self, norient=4):
        self.norient = int(norient)
        self._pyrcls = None
        self._make = None
        self._recon = None
        self._states = None
        self._cfg = None
        self._a1 = None
        self._a2 = None

        # Lazy import, da app dela tudi brez pyrtools
        try:
            import pyrtools as pt
            from pyrtools.pyramids import SteerablePyramidFreq
            self._pyrcls = SteerablePyramidFreq
            # pyrtools 1.x ima rekonstrukcijo kot metodo instance: obj.recon_pyr(coeffs)
            # ali obj.reconstruct; poskusili bomo v runtime-u.
        except Exception as e:
            self._pyrcls = None

    def _init_iir(self, low, high, fps):
        # enako kot v tvojem “originalu”: difference-of-exponentials
        self._a1 = float(np.exp(-2.0 * np.pi * low  / fps))
        self._a2 = float(np.exp(-2.0 * np.pi * high / fps))

    def reset(self, levels, low, high, fps, shape_hw):
        if self._pyrcls is None:
            raise RuntimeError("SteerableBackend requires 'pyrtools' (pip install pyrtools)")

        H, W = shape_hw
        self._cfg = dict(levels=int(levels), low=float(low), high=float(high), fps=float(fps), shape=(H, W))
        self._init_iir(low, high, fps)
        self._states = None  # per-band IIR stanja bomo prvič ustvarili po dekompoziciji

    def _ensure_states(self, coeffs):
        """Ustvari per-band IIR stanja glede na slovar koeficientov."""
        if self._states is not None:
            return
        self._states = {}
        for key, band in coeffs.items():
            if not isinstance(band, np.ndarray):
                continue
            # stanja za fazo: low-pass in high-pass integratorji
            shp = band.shape
            self._states[key] = {
                "phi_prev": np.zeros(shp, np.float32),
                "lp": np.zeros(shp, np.float32),
                "hp": np.zeros(shp, np.float32),
            }

    def _bandpass_phase(self, key, phi):
        st = self._states[key]
        # IIR integratorji
        lp = self._a1 * st["lp"] + (1.0 - self._a1) * phi
        hp = self._a2 * st["hp"] + (1.0 - self._a2) * phi
        st["lp"], st["hp"] = lp, hp
        return (hp - lp)

    def _reconstruct(self, pyr_obj, coeffs_new):
        # Rekonstrukcija: nekaj verzij ima recon_pyr, nekatere reconstruct
        if hasattr(pyr_obj, "recon_pyr"):
            return pyr_obj.recon_pyr(coeffs_new)
        if hasattr(pyr_obj, "reconstruct"):
            return pyr_obj.reconstruct(coeffs_new)
        raise RuntimeError("Unsupported pyrtools version: missing recon method.")

    def magnify(self, gray01, alpha):
        if self._cfg is None:
            raise RuntimeError("SteerableBackend not initialized (call reset).")
        if self._pyrcls is None:
            raise RuntimeError("pyrtools not available.")

        H, W = self._cfg["shape"]
        if gray01.shape != (H, W):
            # varnostno: prilagodimo vhod dekompoziciji, rekonstruiramo na izhodno velikost
            import cv2
            gray01 = cv2.resize(gray01, (W, H), interpolation=cv2.INTER_AREA)

        # 1) Decomposition
        # order = norient-1 (npr. norient=4 -> order=3)
        order = max(1, self.norient - 1)
        pyr = self._pyrcls(gray01, height=self._cfg["levels"], order=order, is_complex=True)
        coeffs = pyr.pyr_coeffs  # dict: ključi bandov in residualov

        # 2) Init IIR states (prvič)
        self._ensure_states(coeffs)

        # 3) Phase push na vsakem kompleksnem bandu
        coeffs_new = {}
        for key, band in coeffs.items():
            arr = band
            if not isinstance(arr, np.ndarray):
                coeffs_new[key] = arr
                continue
            if np.iscomplexobj(arr):
                A = np.abs(arr)
                phi = np.angle(arr).astype(np.float32)

                dphi = self._bandpass_phase(key, phi)  # IIR band-pass na fazi
                phi2 = phi + float(alpha) * dphi

                arr2 = A * np.exp(1j * phi2)
                coeffs_new[key] = arr2.astype(np.complex64)
            else:
                # realni residui (HP/LP) prenesemo nespremenjene
                coeffs_new[key] = arr

        # 4) Rekonstrukcija
        recon = self._reconstruct(pyr, coeffs_new)
        recon = np.clip(recon, 0.0, 1.0).astype(np.float32)
        return recon


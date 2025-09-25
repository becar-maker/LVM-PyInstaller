import numpy as np, math
class TemporalIIRBandpass:
    def __init__(self, low_hz:float, high_hz:float, fps:float, shape):
        aH = math.exp(-2*math.pi*high_hz/fps)
        aL = math.exp(-2*math.pi*low_hz/fps)
        self.aH=np.float32(aH); self.aL=np.float32(aL)
        self.yH=np.zeros(shape,np.float32); self.yL=np.zeros(shape,np.float32)
        self._initialized=False
    def update(self, x:np.ndarray)->np.ndarray:
        x=x.astype(np.float32)
        if not self._initialized:
            self.yH[...] = x; self.yL[...] = x; self._initialized=True
        self.yH = (1.0-self.aH)*x + self.aH*self.yH
        self.yL = (1.0-self.aL)*x + self.aL*self.yL
        return self.yH - self.yL

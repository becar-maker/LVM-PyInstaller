from abc import ABC, abstractmethod

class MagnifierBackend(ABC):
    @abstractmethod
    def reset(self, levels, low, high, fps, shape_hw):
        ...

    @abstractmethod
    def magnify(self, gray01, alpha):
        """gray01: float32 [0..1], returns float32 [0..1]"""
        ...


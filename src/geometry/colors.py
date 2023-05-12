from __future__ import annotations
from numpy.typing import ArrayLike
import numpy as np

from .array import TrackedArray

# TODO: RGBA?

class Colors(TrackedArray):
    """Storage for RGB colors."""
    def __new__(cls, colors, **kwargs):
        if colors is None: colors = np.empty((0, 3), **kwargs)
        res = np.asanyarray(colors, **kwargs).view(cls)
        if not res.shape[1] == 3: raise ValueError("Colors must be (n, 3)")
        return res
    
    @property
    def hsv(self) -> Colors:
        """`np.ndarray (n, 3)` of HSV colors."""
        maxc = self.max(axis=1)
        minc = self.min(axis=1)
        rangec = (maxc-minc)
        v = maxc
        s = np.where(minc == maxc, 0, rangec / maxc)
        rc = np.where(minc == maxc, 0, (maxc-self[:, 0]) / rangec)
        gc = np.where(minc == maxc, 0, (maxc-self[:, 1]) / rangec)
        bc = np.where(minc == maxc, 0, (maxc-self[:, 2]) / rangec)
        h = np.where(self[:, 0] == maxc, bc-gc, np.where(self[:, 1] == maxc, 2.0+rc-bc, 4.0+gc-rc))
        h = (h/6.0) % 1.0
        return np.stack([h, s, v], axis=1).view(Colors)

    @classmethod
    def from_hsv(cls, hsv: ArrayLike) -> Colors:
        """Construct from an ArrayLike of HSV colors."""
        hsv = np.asarray(hsv)
        h, s, v = hsv.T
        i = (h*6.0).astype(int)
        f = (h*6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s*f)
        t = v * (1.0 - s*(1.0-f))
        i = i % 6
        return np.stack([v*(i==0) + q*(i==1) + p*(i==2) + p*(i==3) + t*(i==4) + v*(i==5),
                         t*(i==0) + v*(i==1) + v*(i==2) + q*(i==3) + p*(i==4) + p*(i==5),
                         p*(i==0) + p*(i==1) + t*(i==2) + v*(i==3) + v*(i==4) + q*(i==5)], axis=1).view(Colors)

    @property
    def hsl(self) -> np.ndarray:
        """`np.ndarray (n, 3)` of HSL colors."""
        maxc = self.max(axis=1)
        minc = self.min(axis=1)
        rangec = (maxc-minc)
        l = (maxc+minc)/2.0
        s = np.where(minc == maxc, 0, np.where(l < 0.5, rangec / (maxc+minc), rangec / (2.0-maxc-minc)))
        rc = np.where(minc == maxc, 0, (maxc-self[:, 0]) / rangec)
        gc = np.where(minc == maxc, 0, (maxc-self[:, 1]) / rangec)
        bc = np.where(minc == maxc, 0, (maxc-self[:, 2]) / rangec)
        h = np.where(self[:, 0] == maxc, bc-gc, np.where(self[:, 1] == maxc, 2.0+rc-bc, 4.0+gc-rc))
        h = (h/6.0) % 1.0
        return np.stack([h, s, l], axis=1).view(Colors)
    
    @classmethod
    def from_hsl(cls, hsl: ArrayLike) -> Colors:
        """Construct `Colors` from an ArrayLike of HSL colors."""
        hsl = np.asarray(hsl)
        h, s, l = hsl.T
        i = (h*6.0).astype(int)
        f = (h*6.0) - i
        p = l * (1.0 - s)
        q = l * (1.0 - s*f)
        t = l * (1.0 - s*(1.0-f))
        i = i % 6
        return np.stack([l*(i==0) + q*(i==1) + p*(i==2) + p*(i==3) + t*(i==4) + l*(i==5),
                         t*(i==0) + l*(i==1) + l*(i==2) + q*(i==3) + p*(i==4) + p*(i==5),
                         p*(i==0) + p*(i==1) + t*(i==2) + l*(i==3) + l*(i==4) + q*(i==5)], axis=1).view(Colors)

    @property
    def hex(self) -> np.ndarray:
        """`np.ndarray (n,)` of hex strings."""
        colors = (self * 255).astype(int)
        _hex = (colors * np.array([65536, 256, 1])[None, None, :]).sum(axis=2)
        return np.vectorize(lambda x: "#" + hex(x)[2:].zfill(6))(_hex).flatten()
    
    @classmethod
    def from_hex(cls, hex: ArrayLike) -> Colors:
        """Construct `Colors` from an ArrayLike of hex strings."""
        hex = np.asarray(hex).flatten()
        _hex = np.vectorize(lambda x: int(x, 16))([x[1:] for x in hex])
        colors = (_hex[:, None] // np.array([65536, 256, 1])) % 256
        return (colors / 255).view(Colors)

    @classmethod
    def solid_like(cls, arr: ArrayLike, color=None):
        """Construct `Colors` with the same shape as `arr` with a solid color.
        If `color` is not specified, a random hue is chosen."""
        if color is None:
            color = cls.from_hsv(np.array([np.random.rand(), 0.7, 1.0]))
        return cls(np.full_like(arr, color))
    
    @classmethod
    def random(cls, n=1):
        return cls(np.random.rand(n, 3))
    
    def normalize(self) -> Colors:
        return ((self - self.min()) / (self.max() - self.min())).view(Colors)
    
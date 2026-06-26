"""
masks.py - geometric masks for defining portal regions

Each mask implements three operations:
  __call__(X, Y)     boolean array (for vectorized rendering)
  __contains__(pt)   bool (for point-in-mask checks on mouse click)
  translate(dx, dy)  moves the mask
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Tuple


class Mask(ABC):
    """Abstract mask. Defines a geometric region in the simulation"""

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Returns a boolean array with the same shape as X and Y"""

    @abstractmethod
    def __contains__(self, point: Tuple[float, float]) -> bool:
        """Checks whether a scalar point (x, y) belongs to the mask"""

    @abstractmethod
    def translate(self, dx: float, dy: float) -> None:
        """Shifts the mask by vector (dx, dy)"""


@dataclass
class LineMask(Mask):
    """Line segment from point (x1, y1) to point (x2, y2) with the given thickness"""

    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 0.5

    def __post_init__(self) -> None:
        self._update_cache()

    def _update_cache(self) -> None:
        self._dx = self.x2 - self.x1
        self._dy = self.y2 - self.y1
        self._len_sq = max(self._dx ** 2 + self._dy ** 2, 1e-12)

    def _dist_to_segment(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        vx = X - self.x1
        vy = Y - self.y1
        t = np.clip((vx * self._dx + vy * self._dy) / self._len_sq, 0.0, 1.0)
        px = self.x1 + t * self._dx
        py = self.y1 + t * self._dy
        return np.hypot(X - px, Y - py)

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self._dist_to_segment(X, Y) <= self.thickness

    def __contains__(self, point: Tuple[float, float]) -> bool:
        x, y = point
        buf = self.thickness + 2.0
        if not (min(self.x1, self.x2) - buf <= x <= max(self.x1, self.x2) + buf
                and min(self.y1, self.y2) - buf <= y <= max(self.y1, self.y2) + buf):
            return False
        return float(self._dist_to_segment(np.array(x), np.array(y))) <= self.thickness + 1.0

    def translate(self, dx: float, dy: float) -> None:
        self.x1 += dx
        self.y1 += dy
        self.x2 += dx
        self.y2 += dy
        self._update_cache()


@dataclass
class CircleMask(Mask):
    """Circular mask centered at (cx, cy) with radius radius"""

    cx: float
    cy: float
    radius: float

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (X - self.cx) ** 2 + (Y - self.cy) ** 2 <= self.radius ** 2

    def __contains__(self, point: Tuple[float, float]) -> bool:
        x, y = point
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2

    def translate(self, dx: float, dy: float) -> None:
        self.cx += dx
        self.cy += dy


@dataclass
class RectangleMask(Mask):
    """Rectangular mask defined by ranges [x_min, x_max] by [y_min, y_max]"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        import math
        xlo = math.floor(self.x_min);  xhi = math.ceil(self.x_max)
        ylo = math.floor(self.y_min);  yhi = math.ceil(self.y_max)

        if xhi <= xlo: xhi = xlo + 1
        if yhi <= ylo: yhi = ylo + 1
        return (X >= xlo) & (X < xhi) & (Y >= ylo) & (Y < yhi)

    def __contains__(self, point: Tuple[float, float]) -> bool:
        x, y = point

        xlo = math.floor(self.x_min);  xhi = math.ceil(self.x_max)
        ylo = math.floor(self.y_min);  yhi = math.ceil(self.y_max)
        if xhi <= xlo: xhi = xlo + 1
        if yhi <= ylo: yhi = ylo + 1
        return xlo <= x < xhi and ylo <= y < yhi

    def translate(self, dx: float, dy: float) -> None:
        self.x_min += dx
        self.x_max += dx
        self.y_min += dy
        self.y_max += dy


@dataclass
class PolygonMask(Mask):
    """
    Mask in the shape of an arbitrary polygon

    Args:
        vertices: list of vertices in traversal order [(x0,y0), (x1,y1), ...]
    """

    vertices: List[Tuple[float, float]]

    def __post_init__(self) -> None:
        self._vx = np.array([v[0] for v in self.vertices], dtype=float)
        self._vy = np.array([v[1] for v in self.vertices], dtype=float)

    def _point_in_polygon(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        n = len(self._vx)
        inside = np.zeros(px.shape, dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = self._vx[i], self._vy[i]
            xj, yj = self._vx[j], self._vy[j]
            cond = ((yi > py) != (yj > py)) & (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
            )
            inside ^= cond
            j = i
        return inside

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self._point_in_polygon(X.astype(float), Y.astype(float))

    def __contains__(self, point: Tuple[float, float]) -> bool:
        x, y = float(point[0]), float(point[1])
        return bool(self._point_in_polygon(np.array([x]), np.array([y]))[0])

    def translate(self, dx: float, dy: float) -> None:
        self._vx += dx
        self._vy += dy
        self.vertices = list(zip(self._vx.tolist(), self._vy.tolist()))


@dataclass
class FunctionMask(Mask):
    """
    Mask defined by a function

    Examples:
        FunctionMask("(x - 50)**2 + (y - 50)**2 < 20**2")  # circle
        FunctionMask("np.abs(x - 50) < 10")  # strip

    Args:
        expression: expression string
        offset_x, offset_y: coordinate system offset
    """

    expression: str
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self) -> None:
        self._compiled = compile(self.expression, "<FunctionMask>", "eval")

    def _eval(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return eval(self._compiled, {"np": np}, {"x": x - self.offset_x, "y": y - self.offset_y})

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.asarray(self._eval(X, Y), dtype=bool)

    def __contains__(self, point: Tuple[float, float]) -> bool:
        x, y = point
        return bool(self._eval(np.float64(x), np.float64(y)))

    def translate(self, dx: float, dy: float) -> None:
        self.offset_x += dx
        self.offset_y += dy

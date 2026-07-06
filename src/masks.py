"""
masks.py - geometric masks for defining portal regions

Each mask implements these operations:
  __call__(X, Y)     boolean array (for vectorized rendering)
  __contains__(pt)   bool (for point-in-mask checks on mouse click)
  translate(dx, dy)  moves the mask by a relative offset
  center             (x, y) of the mask's reference point
  set(x, y)          moves the mask to an absolute position
  size()             (width, height) of the mask's bounding box, or None
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


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

    @property
    @abstractmethod
    def center(self) -> Tuple[float, float]:
        """Returns the (x, y) reference point of the mask"""

    def set(self, x: float, y: float) -> None:
        """Moves the mask so that its center is at the absolute point (x, y)"""
        cx, cy = self.center
        self.translate(x - cx, y - cy)

    def size(self) -> Optional[Tuple[float, float]]:
        """Returns the (width, height) of the mask's bounding box, or None if
        not analytically computable"""
        return None


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

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def size(self) -> Optional[Tuple[float, float]]:
        return (abs(self.x2 - self.x1) + 2 * self.thickness,
                abs(self.y2 - self.y1) + 2 * self.thickness)


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

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def size(self) -> Optional[Tuple[float, float]]:
        return (2 * self.radius, 2 * self.radius)


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

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def size(self) -> Optional[Tuple[float, float]]:
        return (self.x_max - self.x_min, self.y_max - self.y_min)


@dataclass
class PointMask(Mask):
    """Single-pixel mask at (x, y)"""

    x: float
    y: float

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (X == int(round(self.x))) & (Y == int(round(self.y)))

    def __contains__(self, point: Tuple[float, float]) -> bool:
        px, py = point
        return abs(px - self.x) < 1.0 and abs(py - self.y) < 1.0

    def translate(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def size(self) -> Optional[Tuple[float, float]]:
        return (1.0, 1.0)


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

    @property
    def center(self) -> Tuple[float, float]:
        return (float(np.mean(self._vx)), float(np.mean(self._vy)))

    def size(self) -> Optional[Tuple[float, float]]:
        return (float(self._vx.max() - self._vx.min()),
                float(self._vy.max() - self._vy.min()))


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

    @property
    def center(self) -> Tuple[float, float]:
        return (self.offset_x, self.offset_y)

    def size(self) -> Optional[Tuple[float, float]]:
        # Not analytically computable from an arbitrary expression - kept as
        # an explicit override (rather than relying on the base default) so
        # it's clear this is intentional and not a missing implementation.
        return None


@dataclass
class ArrayMask(Mask):
    """
    Mask backed by an explicit boolean grid shaped exactly like the
    simulation's (H, W) coordinate grid. Represents irregular/composite
    regions that no geometric formula can express - e.g. a MaterialObject
    mid-teleport, split between a source and destination portal.
    """

    grid: np.ndarray  # bool, shape (H, W): axis 0 = Y/row, axis 1 = X/col

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=bool)
        self._accum_dx: float = 0.0  # sub-pixel translate() accumulator
        self._accum_dy: float = 0.0

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        expected_shape = np.broadcast(X, Y).shape
        if self.grid.shape != expected_shape:
            raise ValueError(
                f"ArrayMask grid shape {self.grid.shape} does not match "
                f"simulation grid shape {expected_shape}")
        return self.grid

    def __contains__(self, point: Tuple[float, float]) -> bool:
        x, y = point
        xi, yi = int(round(x)), int(round(y))
        h, w = self.grid.shape
        if not (0 <= yi < h and 0 <= xi < w):
            return False
        return bool(self.grid[yi, xi])

    def translate(self, dx: float, dy: float) -> None:
        self._accum_dx += dx
        self._accum_dy += dy
        shift_x = int(round(self._accum_dx))
        shift_y = int(round(self._accum_dy))
        if shift_x == 0 and shift_y == 0:
            return
        self._accum_dx -= shift_x
        self._accum_dy -= shift_y
        self.grid = ArrayMask._shift_grid(self.grid, shift_x, shift_y)

    @staticmethod
    def _shift_grid(grid: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
        """Shifts a boolean grid by (shift_x, shift_y) pixels, zero-filling
        vacated cells (no wraparound, unlike np.roll)"""
        h, w = grid.shape
        out = np.zeros_like(grid)

        if shift_x >= 0:
            sx_src, sx_dst = slice(0, w - shift_x), slice(shift_x, w)
        else:
            sx_src, sx_dst = slice(-shift_x, w), slice(0, w + shift_x)
        if shift_y >= 0:
            sy_src, sy_dst = slice(0, h - shift_y), slice(shift_y, h)
        else:
            sy_src, sy_dst = slice(-shift_y, h), slice(0, h + shift_y)

        if sx_dst.start >= sx_dst.stop or sy_dst.start >= sy_dst.stop:
            return out  # shifted entirely off-grid
        out[sy_dst, sx_dst] = grid[sy_src, sx_src]
        return out

    @property
    def center(self) -> Tuple[float, float]:
        ys, xs = np.nonzero(self.grid)
        if len(xs) == 0:
            # Empty mask: no meaningful center. Fall back to the grid's
            # center rather than raising - callers (np.any guards) already
            # skip empty masks before this would matter physically.
            h, w = self.grid.shape
            return (w / 2.0, h / 2.0)
        return (float(np.mean(xs)), float(np.mean(ys)))

    def size(self) -> Optional[Tuple[float, float]]:
        ys, xs = np.nonzero(self.grid)
        if len(xs) == 0:
            return (0.0, 0.0)
        return (float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))

    def set(self, region: np.ndarray) -> None:
        """Replaces the mask's content with `region` (bool array, must match
        the stored grid's shape). Overrides the base class's position-based
        `set(x, y)` for this subclass only - see masks.py module docstring"""
        region = np.asarray(region, dtype=bool)
        if region.shape != self.grid.shape:
            raise ValueError(
                f"region shape {region.shape} does not match mask grid "
                f"shape {self.grid.shape}")
        self.grid = region
        self._accum_dx = 0.0
        self._accum_dy = 0.0

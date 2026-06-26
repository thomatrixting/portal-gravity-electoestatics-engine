import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple


class ColorMapper(ABC):
    """Abstract color mapper: values -> RGB uint8 array"""

    @abstractmethod
    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Returns a uint8 array of RGB colors"""


class SingleColorMapper(ColorMapper):
    """Mapper that returns the same color for all values"""

    def __init__(self, color: Tuple[int, int, int]) -> None:
        self.color = np.array(color, dtype=np.uint8)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        result = np.empty(values.shape + (3,), dtype=np.uint8)
        result[:] = self.color
        return result


class GradientColorMapper(ColorMapper):
    """
    Linear gradient between control points

    Args:
        points: list of (value, (R, G, B)) pairs, sorted by value
    """

    def __init__(self, points: List[Tuple[float, Tuple[int, int, int]]]) -> None:
        self.points = sorted(points, key=lambda p: p[0])
        self.values = np.array([p[0] for p in self.points], dtype=np.float32)
        self.colors = np.array([p[1] for p in self.points], dtype=np.float32)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.clip(values, self.values[0], self.values[-1]).astype(np.float32)
        idx = np.searchsorted(self.values, v, side="right") - 1
        idx = np.clip(idx, 0, len(self.points) - 2)

        v0 = self.values[idx]
        v1 = self.values[idx + 1]
        t = (v - v0) / np.where(v1 != v0, v1 - v0, 1.0)
        t = t[..., np.newaxis]  # broadcast for RGB

        interpolated = self.colors[idx] * (1.0 - t) + self.colors[idx + 1] * t
        return np.clip(interpolated, 0, 255).astype(np.uint8)


# region Color schemes

def colormap_plasma() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (13, 8, 135)),  # dark violet
        (0.25, (84, 2, 163)),  # violet
        (0.5, (139, 10, 165)),  # magenta
        (0.75, (211, 97, 107)),  # pinkish orange
        (1.0, (240, 249, 33)),  # yellow
    ])


def colormap_electric() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (0, 0, 0)),  # black
        (0.3, (20, 20, 180)),  # dark blue
        (0.6, (70, 130, 255)),  # blue
        (0.85, (180, 220, 255)),  # light blue
        (1.0, (255, 255, 255)),  # white
    ])



def colormap_fire() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (0, 0, 0)),  # black
        (0.25, (128, 0, 0)),  # dark red
        (0.5, (255, 69, 0)),  # orange-red
        (0.75, (255, 215, 0)),  # golden
        (1.0, (255, 255, 255)),  # white
    ])


def default_color_mapper() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (0, 0, 139)),  # dark blue
        (1.0, (0, 150, 255)),  # bright blue
        (2.5, (50, 205, 50)),  # green
        (5.0, (255, 215, 0)),  # yellow
        (7.5, (255, 140, 0)),  # orange
        (10.0, (255, 60, 60)),  # red (no white, to avoid burning out)
    ])


def extra_mapper() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (0, 0, 139)),  # dark blue
        (0.25, (30, 144, 255)),  # blue
        (0.5, (0, 200, 200)),  # turquoise
        (0.75, (255, 220, 0)),  # yellow
        (1.0, (220, 20, 60)),  # red
    ])


def default_potential_mapper() -> GradientColorMapper:
    return GradientColorMapper([
        (0.0, (30, 30, 200)),   # blue
        (0.5, (60, 200, 80)),   # green
        (1.0, (210, 30, 30)),   # red
    ])
# endregion


# List of color schemes
COLOR_SCHEMES = {
    "Default": default_color_mapper,
    "Potential": default_potential_mapper,
    "Extra": extra_mapper,
    "Plasma": colormap_plasma,
    "Electric": colormap_electric,
    "Fire": colormap_fire,
}

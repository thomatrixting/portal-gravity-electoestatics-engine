import numpy as np

class BoundaryMesh:
    def __init__(self, mask, W, H, potential_value, dx=1.0):
        """
        mask: cualquier Mask del repo (CircleMask, RectangleMask, etc.)
        W, H: dimensiones de la grilla
        potential_value: phi prescrito en esa superficie
        dx: tamaño de celda
        """
        self.phi = potential_value
        
        # Obtener el array booleano usando la convención del repo
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        interior = mask(X, Y)  # bool 2D de shape (H, W)
        
        self.segments = self._extract_boundary(interior, potential_value, dx)

    def _extract_boundary(self, interior, phi, dx):
        segments = []
        rows, cols = interior.shape
        for y in range(rows):
            for x in range(cols):
                if not interior[y, x]:
                    continue
                neighbors = [
                    (y-1, x), (y+1, x), (y, x-1), (y, x+1)
                ]
                is_boundary = any(
                    not (0 <= ny < rows and 0 <= nx < cols and interior[ny, nx])
                    for ny, nx in neighbors
                )
                if is_boundary:
                    segments.append((x * dx, y * dx, dx, phi))
        return segments
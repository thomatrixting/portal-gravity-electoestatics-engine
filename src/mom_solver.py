import numpy as np

class MOMSolver2D:
    def __init__(self, boundary_meshes):
        self.segments = []
        for mesh in boundary_meshes:
            self.segments.extend(mesh.segments)
        self.N = len(self.segments)
        self.sigma = None

    def build_and_solve(self):
        N = self.N
        A = np.zeros((N, N))
        b = np.zeros(N)

        for i, (xi, yi, li, phi_i) in enumerate(self.segments):
            b[i] = phi_i
            for j, (xj, yj, lj, _) in enumerate(self.segments):
                if i == j:
                    # Auto-influencia adimensional
                    A[i, j] = lj * (1.0 - np.log(lj / 2.0))
                else:
                    r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    A[i, j] = -np.log(r) * lj

        self.sigma = np.linalg.solve(A, b)
        return self.sigma

    def compute_phi_grid(self, grid_x, grid_y):
        phi = np.zeros_like(grid_x, dtype=float)
        for k, (xj, yj, lj, _) in enumerate(self.segments):
            r = np.sqrt((grid_x - xj)**2 + (grid_y - yj)**2 + 1e-10)
            phi += -np.log(r) * lj * self.sigma[k]
        return phi
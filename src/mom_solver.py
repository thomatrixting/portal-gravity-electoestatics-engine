import numpy as np

class MOMSolver2D:
    def __init__(self, boundary_meshes, coupled_pairs=None):
        """
        boundary_meshes: lista de BoundaryMesh con phi fijo
        coupled_pairs: lista de grupos; cada grupo es un iterable de
                       BoundaryMesh (2, 3, o más) que comparten un mismo
                       phi desconocido (ej. CouplePortal o MultiPortal)
        """
        self.segments = []
        self.segment_group = []  # índice de grupo para acoplados

        # Segmentos con phi fijo
        for mesh in boundary_meshes:
            for seg in mesh.segments:
                self.segments.append(seg)
                self.segment_group.append(None)  # phi conocido

        # Segmentos acoplados: phi desconocido compartido entre N mallas
        self.coupled_pairs = coupled_pairs or []
        self.coupled_offsets = []  # dónde empieza cada grupo en self.segments
        for pair_idx, group in enumerate(self.coupled_pairs):
            offset = len(self.segments)
            for mesh in group:
                for seg in mesh.segments:
                    self.segments.append((seg[0], seg[1], seg[2], None))
                    self.segment_group.append(pair_idx)
            self.coupled_offsets.append(offset)

        self.N = len(self.segments)
        self.n_pairs = len(self.coupled_pairs)
        self.sigma = None
        self.phi_coupled = None

    def build_and_solve(self):
        N = self.N
        P = self.n_pairs
        size = N + P
        A = np.zeros((size, size))
        b = np.zeros(size)

        for i, (xi, yi, li, phi_i) in enumerate(self.segments):
            for j, (xj, yj, lj, _) in enumerate(self.segments):
                if i == j:
                    A[i, j] = lj * (1.0 - np.log(lj / 2.0))
                else:
                    r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    A[i, j] = -np.log(r) * lj

            if phi_i is not None:
                b[i] = phi_i
            else:
                pair_idx = self.segment_group[i]
                A[i, N + pair_idx] = -1.0
                b[i] = 0.0

        for pair_idx, offset in enumerate(self.coupled_offsets):
            row = N + pair_idx
            for j in range(offset, len(self.segments)):
                if self.segment_group[j] == pair_idx:
                    A[row, j] = self.segments[j][2]
            b[row] = 0.0

        solution = np.linalg.solve(A, b)
        self.sigma = solution[:N]
        self.phi_coupled = solution[N:]
        return self.sigma

    def compute_phi_grid(self, grid_x, grid_y):
        phi = np.zeros_like(grid_x, dtype=float)
        for k, (xj, yj, lj, _) in enumerate(self.segments):
            r = np.sqrt((grid_x - xj)**2 + (grid_y - yj)**2 + 1e-10)
            phi += -np.log(r) * lj * self.sigma[k]
        return phi
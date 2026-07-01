import numpy as np
import matplotlib.pyplot as plt
from masks import CircleMask, RectangleMask
from mom_mesh import BoundaryMesh
from mom_solver import MOMSolver2D

W, H = 60, 60  # grilla pequeña para que sea rápido

# Dos placas (arriba phi=1, abajo phi=0) y un conductor en el centro
mesh_top  = BoundaryMesh(RectangleMask(0, W, 0, 1),      W, H, potential_value=1.0)
mesh_bot  = BoundaryMesh(RectangleMask(0, W, H-1, H),    W, H, potential_value=0.0)
mesh_cond = BoundaryMesh(CircleMask(cx=30, cy=30, radius=8), W, H, potential_value=0.5)

print(f"Segmentos placa top:    {len(mesh_top.segments)}")
print(f"Segmentos placa bottom: {len(mesh_bot.segments)}")
print(f"Segmentos conductor:    {len(mesh_cond.segments)}")
print(f"Total N: {len(mesh_top.segments) + len(mesh_bot.segments) + len(mesh_cond.segments)}")

solver = MOMSolver2D([mesh_top, mesh_bot, mesh_cond])
sigma = solver.build_and_solve()
print(f"sigma min={sigma.min():.4f}  max={sigma.max():.4f}")

# Reconstruir y visualizar
xs = np.arange(W, dtype=float)
ys = np.arange(H, dtype=float)
grid_x, grid_y = np.meshgrid(xs, ys)
phi = solver.compute_phi_grid(grid_x, grid_y)

plt.figure(figsize=(6,5))
plt.contourf(grid_x, grid_y, phi, levels=30, cmap='plasma')
plt.colorbar(label='φ')
plt.title('MOM - Potencial reconstruido')
plt.savefig('test_mom_output.png')
print("Imagen guardada en test_mom_output.png")
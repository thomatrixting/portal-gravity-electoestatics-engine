"""
This file contains example scenes that demonstrate some of this program's features
"""


from simulation import Simulation
from portals import *
from masks import *


def _anchors(sim_width, sim_height):
    """Special preset with potential anchors so gravity behaves like on Earth"""
    return [
        PotentialAnchor(RectangleMask(0, sim_width, 0, 1), 1.0),
        PotentialAnchor(RectangleMask(0, sim_width, sim_height-1, sim_height), 0.0),
    ]

def example_portal_on_capacitor() -> Simulation:
    """one paralel portals between a capacitor that would be the roof, to another portal far away from the other portal"""
    W,H = 400,200
    px_sclae = 4

    cap_length = H/2
    portal_lenght= H/4

    cap_delta_with_top = (H - cap_length)/2
    portal_delta_with_top = (H - portal_lenght)/2

    #define the capacitor
    capacitor = [
        PotentialAnchor(RectangleMask(W/8,W/8,cap_delta_with_top,cap_delta_with_top + cap_length),1.0),
        PotentialAnchor(RectangleMask(3*W/8,W/8,cap_delta_with_top,cap_delta_with_top + cap_length),0)
    ]

    p1 = Portal(RectangleMask(W/4, W/4, portal_delta_with_top, portal_delta_with_top + portal_lenght), color=(255, 153, 0))
    p2 = Portal(RectangleMask(3*W/4, 3*W/4, portal_delta_with_top, portal_delta_with_top + portal_lenght), color=(0, 204, 255))

    return Simulation(
        *capacitor, CouplePortal(p1,p2),
        sim_width=W, sim_height=H,
        px_scale=px_sclae,
        iterations_per_frame=2000,
        sor_omega=1.7,
        isoline_count=50
    )


def example_couple_portals() -> Simulation:
    """Two parallel portals"""
    #W, H = 120, 120
    W,H = 200,250
    p1 = Portal(RectangleMask(40, 80, 40, 40), color=(255, 153, 0))
    p2 = Portal(RectangleMask(40, 80, 80, 80), color=(0, 204, 255))

    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=4,
        iterations_per_frame=40,
        sor_omega=1.7,
    )



def example_advanced() -> Simulation:
    """FixedPotential + object"""
    W, H = 120, 100
    top = FixedPotentialPortal(RectangleMask(20, 100, 30, 33), 0.8, (255, 80, 80))
    bot = FixedPotentialPortal(RectangleMask(20, 100, 67, 70), 0.2, (80, 120, 255))
    obs = MaterialObject(CircleMask(60, 50, 8),
                         color=(80, 220, 120), pinned=True, label="Conductor")
    return Simulation(
        top, bot, obs,
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=60,
        sor_omega=1.75,
    )


def example_couple_circles() -> Simulation:
    """Two circular portals"""
    W, H = 120, 100
    p1 = Portal(CircleMask(cx=35, cy=50, radius=10), color=(255, 100, 50))
    p2 = Portal(CircleMask(cx=85, cy=50, radius=10), color=(50, 200, 255))

    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=50,
        sor_omega=1.8,
        isoline_count=15,
    )


def triple_portals() -> Simulation:
    W, H = 120, 100
    p1 = Portal(RectangleMask(40, 60, 30, 30), (255, 0, 0))
    p2 = Portal(RectangleMask(40, 60, 60, 60), (0, 255, 0))
    p3 = Portal(RectangleMask(40, 60, 90, 90), (0, 0, 255))

    return Simulation(
        *_anchors(W, H), MultiPortal((p1, p2, p3)),
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=50,
        sor_omega=1.8,
        isoline_count=15,
    )

def example_mom_conductor():
    from mom_mesh import BoundaryMesh
    from mom_solver import MOMSolver2D
    import numpy as np

    W, H = 120, 120
    dx = 1.0

    # Grilla de coordenadas para reconstrucción
    xs = np.arange(W) * dx
    ys = np.arange(H) * dx
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Defines conductores como arrays booleanos directamente
    # (o los sacas de tus Masks existentes)
    from masks import CircleMask, RectangleMask

    mask_conductor = CircleMask(cx=60, cy=60, radius=15)
    mask_plate_top = RectangleMask(0, W, 0, 1)
    mask_plate_bot = RectangleMask(0, W, H-1, H)

    # Extraes la representación booleana de cada mask
    bool_conductor = mask_conductor.to_array(W, H)  # ajusta al método real
    bool_top       = mask_plate_top.to_array(W, H)
    bool_bot       = mask_plate_bot.to_array(W, H)

    meshes = [
        BoundaryMesh(bool_top,       potential_value=1.0, dx=dx),
        BoundaryMesh(bool_bot,       potential_value=0.0, dx=dx),
        BoundaryMesh(bool_conductor, potential_value=0.5, dx=dx),
    ]

    solver = MOMSolver2D(meshes)
    solver.build_and_solve()
    phi_grid = solver.compute_phi_grid(grid_x, grid_y)

    # phi_grid es un numpy array 2D listo para visualizar
    # Puedes pasarlo a Simulation como campo inicial o visualizarlo con matplotlib
    return phi_grid, solver.sigma

def example_mom() -> Simulation:
    W, H = 80, 80
    top    = PotentialAnchor(RectangleMask(0, W, 0, 1),      1.0)
    bottom = PotentialAnchor(RectangleMask(0, W, H-1, H),    0.0)
    cond   = PotentialAnchor(CircleMask(W//2, H//2, 10),     0.5)  # conductor como anchor
    return Simulation(
        top, bottom, cond,
        sim_width=W, sim_height=H,
        px_scale=5,
        solver_mode="mom",
        show_isolines=True,
    )

def example_mom_couple() -> Simulation:
    W, H = 80, 80
    top    = PotentialAnchor(RectangleMask(0, W, 0, 1),      1.0)
    bottom = PotentialAnchor(RectangleMask(0, W, H-1, H),    0.0)
    # Portales horizontales: uno arriba y uno abajo
    p1 = Portal(RectangleMask(20, 60, 20, 21), (255, 153, 0))
    p2 = Portal(RectangleMask(20, 60, 59, 60), (0, 204, 255))
    return Simulation(
        top, bottom, CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=5,
        solver_mode="mom",
        show_isolines=True,
    )

def example_mom_carga_entre_portales() -> Simulation:
    """Carga flotante entre dos portales, sin campo externo"""
    W, H = 80, 80
    # Sin placas top/bottom — solo los portales y la carga
    p1 = Portal(RectangleMask(10, 11, 15, 65), (255, 153, 0))
    p2 = Portal(RectangleMask(69, 70, 15, 65), (0, 204, 255))
    carga = PotentialAnchor(CircleMask(W//2, H//2, 6), 1.0)
    return Simulation(
        CouplePortal(p1, p2), carga,
        sim_width=W, sim_height=H,
        px_scale=5,
        solver_mode="mom",
        show_isolines=True,
    )


def example_mom_cargas_afuera() -> Simulation:
    """Dos portales verticales con cargas a los lados externos"""
    W, H = 80, 80
    top    = PotentialAnchor(RectangleMask(0, W, 0, 1),   0.0)
    bottom = PotentialAnchor(RectangleMask(0, W, H-1, H), 0.0)
    # Portales verticales en el centro
    p1 = Portal(RectangleMask(35, 36, 20, 60), (255, 153, 0))
    p2 = Portal(RectangleMask(44, 45, 20, 60), (0, 204, 255))
    # Cargas a los lados externos de los portales
    carga_izq = PotentialAnchor(CircleMask(15, H//2, 5), 0.9)
    carga_der = PotentialAnchor(CircleMask(65, H//2, 5), 0.1)
    return Simulation(
        CouplePortal(p1, p2), carga_izq, carga_der,
        sim_width=W, sim_height=H,
        px_scale=5,
        solver_mode="mom",
        show_isolines=True,
    )
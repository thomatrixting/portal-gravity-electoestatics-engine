"""
This file contains example scenes that demonstrate some of this program's features
"""


from simulation import Simulation
from portals import *
from masks import *
from test_charge import TestCharge


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

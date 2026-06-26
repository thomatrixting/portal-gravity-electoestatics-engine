"""
Main simulation script

Objects:
  CouplePortal(p1, p2)           pair of portals with equal potential
  FixedPotentialPortal(mask, v)  region with fixed potential
  PotentialAnchor(mask, v)       potential anchor

Masks:
  RectangleMask(x_min, x_max, y_min, y_max)
  CircleMask(cx, cy, radius)
  PointMask(x, y)
  LineMask(x1, y1, x2, y2, thickness)
  PolygonMask([(x0,y0), ...])
  FunctionMask("expression(x, y)")

Controls:
  M - toggle display mode: gravitational acceleration / potential
  V - vectors on/off
  I - isolines on/off

  Drag - drag any portal with the mouse
  SIM tab - render and physics parameters
  SCENE tab - scene objects, add, presets, inspector

See README.md for details
"""


from scenes import *


def main() -> None:
    sim = example_portal_on_capacitor()
    #sim = example_couple_portals()  # Starting scene
    #sim = triple_portals()
    sim.run()


if __name__ == "__main__":
    main()

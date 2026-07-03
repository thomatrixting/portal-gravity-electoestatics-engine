"""
test_charge.py - test (probe) charges for the electrostatic portal simulation

A TestCharge is a point charge with q -> 0 in spirit: it FEELS the field
produced by the capacitor / portals / conductors, but never CONTRIBUTES to
it. It is therefore deliberately kept OUTSIDE of PhysicsEngineEM.field —
the engine must never know test charges exist. The simulation loop is:

    engine.step()                       # solver doesn't know about charges
    for charge in test_charges:
        charge.update(engine, dt)       # charge reads E, moves itself

Physics
-------
Equation of motion (purely electrostatic, no gravity):

    m * d2r/dt2 = q * E(r)            with   E = -grad(V)

Integrator: Velocity Verlet (symplectic -> bounded energy error over long
runs, unlike explicit/semi-implicit Euler). Needs E evaluated twice per
step (at r_n and r_n+1), which is cheap here because grad_x/grad_y are
already precomputed by the engine — no Laplace re-solve involved.

    r_{n+1} = r_n + v_n*dt + 0.5*a_n*dt^2
    a_{n+1} = (q/m) * E(r_{n+1})
    v_{n+1} = v_n + 0.5*(a_n + a_{n+1})*dt

Field sampling: grad_x / grad_y live on the integer grid, but the charge's
position is continuous, so E is bilinearly interpolated from the four
surrounding grid cells (see _bilinear_sample).

***Boundary handling: clamping. If the (would-be) position falls outside
[0, width-1] x [0, height-1], the position is clipped back onto the
nearest valid coordinate and the velocity component pointing further
outward is zeroed (so the charge doesn't keep "pushing" into the wall —
otherwise clamped position + nonzero outward velocity would just re-exit
next step, and the trajectory would look like it's sliding along the
border rather than stopping there).
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class TestCharge:
    """
    A point probe charge that responds to the field but never perturbs it.

    Args:
        x, y     : initial continuous position (grid coordinates, floats)
        vx, vy   : initial velocity components
        charge   : q, the probe's charge (sign matters; magnitude should be
                   small relative to the field's own sources so it stays a
                   "test" charge in the physical sense — this is a modelling
                   choice by the user, not something this class enforces)
        mass     : m, inertial mass. Required even though the problem is
                   purely electrostatic: it's Newton's second law's own
                   constant (a = F/m), unrelated to gravity.
        color    : RGB tuple for rendering
        active   : if False, update() is a no-op (frozen / disabled charge)
        trail_len: number of past positions kept in self.trail, for drawing
                   the trajectory. 0 disables trail recording.
    """

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    charge: float = 1.0
    mass: float = 1.0
    color: Tuple[int, int, int] = (255, 255, 0)
    active: bool = True
    trail_len: int = 200

    # internal state, not passed by the user
    trail: List[Tuple[float, float]] = field(default_factory=list, init=False)
    _ax: float = field(default=0.0, init=False)   # acceleration at current r_n (cached between calls)
    _ay: float = field(default=0.0, init=False)
    _initialized_accel: bool = field(default=False, init=False)

    def __repr__(self) -> str:
        return (f"TestCharge(q={self.charge}, m={self.mass}, "
                f"pos=({self.x:.2f},{self.y:.2f}), "
                f"v=({self.vx:.3f},{self.vy:.3f}))")

    # ------------------------------------------------------------------
    # Field sampling: bilinear interpolation of grad_x / grad_y
    # ------------------------------------------------------------------

    @staticmethod
    def _bilinear_sample(field_arr: np.ndarray, x: float, y: float,
                          width: int, height: int) -> float:
        """
        Bilinearly interpolate field_arr (shape (height, width)) at the
        continuous coordinate (x, y).  Coordinates are clamped to the valid
        range first, so this never indexes out of bounds even if called
        with an (already clamped) boundary position.
        """
        xc = min(max(x, 0.0), width - 1.0001)
        yc = min(max(y, 0.0), height - 1.0001)

        i0 = int(np.floor(xc))
        j0 = int(np.floor(yc))
        i1 = i0 + 1
        j1 = j0 + 1

        tx = xc - i0
        ty = yc - j0

        f00 = field_arr[j0, i0]
        f10 = field_arr[j0, i1]
        f01 = field_arr[j1, i0]
        f11 = field_arr[j1, i1]

        f0 = f00 * (1 - tx) + f10 * tx   # interpolate along x at row j0
        f1 = f01 * (1 - tx) + f11 * tx   # interpolate along x at row j1
        return float(f0 * (1 - ty) + f1 * ty)   # interpolate along y

    def sample_field(self, engine) -> Tuple[float, float]:
        """
        Returns (Ex, Ey) of the engine's E field, bilinearly interpolated
        at this charge's current (x, y). Read-only: does not touch engine.
        """
        Ex = self._bilinear_sample(engine.grad_x, self.x, self.y,
                                    engine.width, engine.height)
        Ey = self._bilinear_sample(engine.grad_y, self.x, self.y,
                                    engine.width, engine.height)
        return Ex, Ey

    # ------------------------------------------------------------------
    # Dynamics: Velocity Verlet integration
    # ------------------------------------------------------------------

    def _acceleration(self, engine, x: float, y: float) -> Tuple[float, float]:
        """a = (q/m) * E(x, y), evaluated at an arbitrary (clamped) position."""
        Ex, Ey = self._sample_at(engine, x, y)
        qm = self.charge / self.mass
        return qm * Ex, qm * Ey

    def _sample_at(self, engine, x: float, y: float) -> Tuple[float, float]:
        Ex = -self._bilinear_sample(engine.grad_x, x, y, engine.width, engine.height)
        Ey = -self._bilinear_sample(engine.grad_y, x, y, engine.width, engine.height)
        return Ex, Ey

    def update(self, engine, dt: float) -> None:
        """
        Advance the charge by one Velocity Verlet step of size dt, reading
        the (already solved) field from `engine`. Does not modify engine
        in any way — pure read access to engine.grad_x / engine.grad_y,
        engine.width, engine.height.
        """
        if not self.active:
            return

        # Acceleration at the start of the step (a_n). Cache it across
        # calls so each step only needs ONE new field evaluation (at the
        # new position) instead of two — standard Velocity Verlet trick.
        if not self._initialized_accel:
            self._ax, self._ay = self._acceleration(engine, self.x, self.y)
            self._initialized_accel = True

        ax_n, ay_n = self._ax, self._ay

        # ---- position update: r_{n+1} = r_n + v_n dt + 0.5 a_n dt^2 ----
        x_new = self.x + self.vx * dt + 0.5 * ax_n * dt * dt
        y_new = self.y + self.vy * dt + 0.5 * ay_n * dt * dt

        # ---- boundary handling: clamping ----
        x_new, y_new, vx_pre, vy_pre = self._clamp_position(
            engine, x_new, y_new, self.vx, self.vy
        )

        # ---- acceleration at the new position: a_{n+1} ----
        ax_np1, ay_np1 = self._acceleration(engine, x_new, y_new)

        # ---- velocity update: v_{n+1} = v_n + 0.5 (a_n + a_{n+1}) dt ----
        vx_new = vx_pre + 0.5 * (ax_n + ax_np1) * dt
        vy_new = vy_pre + 0.5 * (ay_n + ay_np1) * dt

        # if clamping zeroed an outward velocity component, also kill the
        # corresponding *new* velocity if it still points outward (prevents
        # the charge from re-accelerating straight back out through the
        # wall within the same step due to the a_{n+1} contribution)
        vx_new, vy_new = self._reclamp_velocity(engine, x_new, y_new, vx_new, vy_new)

        self.x, self.y = x_new, y_new
        self.vx, self.vy = vx_new, vy_new
        self._ax, self._ay = ax_np1, ay_np1   # becomes a_n for the next call

        self._record_trail()

    # ------------------------------------------------------------------
    # Boundary handling: clamping
    # ------------------------------------------------------------------

    def _clamp_position(self, engine, x: float, y: float,
                         vx: float, vy: float) -> Tuple[float, float, float, float]:
        """
        Clip (x, y) into [0, width-1] x [0, height-1]. If clamping occurred
        on an axis, zero the velocity component that was driving the charge
        further outward on that axis (so it settles at the wall instead of
        sliding along it indefinitely under a tangential push only — normal
        push is removed, tangential push is preserved).
        """
        x_min, x_max = 0.0, engine.width - 1.0
        y_min, y_max = 0.0, engine.height - 1.0

        x_clamped = min(max(x, x_min), x_max)
        y_clamped = min(max(y, y_min), y_max)

        vx_out = vx
        vy_out = vy

        hit_x_low  = x <= x_min and vx < 0
        hit_x_high = x >= x_max and vx > 0
        if hit_x_low or hit_x_high:
            vx_out = 0.0

        hit_y_low  = y <= y_min and vy < 0
        hit_y_high = y >= y_max and vy > 0
        if hit_y_low or hit_y_high:
            vy_out = 0.0

        return x_clamped, y_clamped, vx_out, vy_out

    def _reclamp_velocity(self, engine, x: float, y: float,
                           vx: float, vy: float) -> Tuple[float, float]:
        """Second pass: if position sits exactly on a wall, zero any
        velocity component still pointing outward after the a_{n+1}
        contribution was added."""
        x_min, x_max = 0.0, engine.width - 1.0
        y_min, y_max = 0.0, engine.height - 1.0

        if (x <= x_min and vx < 0) or (x >= x_max and vx > 0):
            vx = 0.0
        if (y <= y_min and vy < 0) or (y >= y_max and vy > 0):
            vy = 0.0
        return vx, vy

    # ------------------------------------------------------------------
    # Trajectory bookkeeping
    # ------------------------------------------------------------------

    def _record_trail(self) -> None:
        if self.trail_len <= 0:
            return
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.trail_len:
            self.trail.pop(0)

    def reset_trail(self) -> None:
        self.trail.clear()

    def kinetic_energy(self) -> float:
        """0.5 * m * |v|^2 — useful to sanity-check Verlet's energy behaviour."""
        return 0.5 * self.mass * (self.vx ** 2 + self.vy ** 2)

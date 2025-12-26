#!/usr/bin/env python3
"""
Compute effective PV geometry from multiple sub-arrays using capacity-weighted
surface normal vectors.

Key ideas:
- Each sub-array has a unit normal n_i (orientation only).
- Weight by capacity P_i (kWp) to get a "capacity vector" contribution P_i * n_i.
- Sum: N = sum_i P_i * n_i
    * direction  n_eff = N / ||N||   -> effective orientation (single equivalent plane)
    * magnitude  P_equiv = ||N||     -> equivalent kWp of that single plane (<= sum P_i)

If you want time-dependent "effective kWp toward the sun", use:
  P_toward_sun(s_hat) = sum_i P_i * max(0, n_i dot s_hat)
(not equal to max(0, N dot s_hat) in general because of the max(0,dot) nonlinearity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
OUTPUT_CONFIG = PROJECT_ROOT / "configs" / "config_effective_target.yaml"


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class SubArray:
    name: str
    azimuth_deg: float  # A: 0=N, 90=E, 180=S, 270=W (per your convention)
    tilt_deg: float     # T: 0=horizontal (normal points up), 90=vertical
    p_kwp: float        # capacity weight (kWp)


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(v: Vec3) -> float:
    return math.sqrt(dot(v, v))


def normalize(v: Vec3) -> Vec3:
    m = norm(v)
    if m == 0.0:
        raise ValueError("Zero vector cannot be normalized.")
    return (v[0] / m, v[1] / m, v[2] / m)


def unit_normal_from_AT(azimuth_deg: float, tilt_deg: float) -> Vec3:
    """
    Forward mapping: (A, T) -> unit normal n using your convention:

        n_x =  sin(T) * sin(A)
        n_y = -sin(T) * cos(A)
        n_z =  cos(T)

    with angles in degrees and trig in radians internally.
    """
    A = deg2rad(azimuth_deg)
    T = deg2rad(tilt_deg)

    nx = math.sin(T) * math.sin(A)
    ny = -math.sin(T) * math.cos(A)
    nz = math.cos(T)

    # analytically already unit length, normalize for numeric safety
    return normalize((nx, ny, nz))


def AT_from_unit_normal(n: Vec3) -> Tuple[float, float]:
    """
    Back-transform: unit normal -> (A, T) using your convention.

        T = arccos(nz)
        A = atan2(nx, -ny)  mapped to [0, 360)
    """
    nx, ny, nz = normalize(n)

    # clamp to avoid acos domain issues from rounding
    nz = max(-1.0, min(1.0, nz))

    T = math.acos(nz)  # radians

    # If T ~ 0, azimuth undefined; choose 0 by convention
    if abs(math.sin(T)) < 1e-12:
        A = 0.0
    else:
        A = math.atan2(nx, -ny)

    return (rad2deg(A) % 360.0, rad2deg(T))


def capacity_vector(arrays: List[SubArray]) -> Vec3:
    """N = sum_i P_i * n_i (NOT normalized)."""
    sx = sy = sz = 0.0
    for a in arrays:
        nx, ny, nz = unit_normal_from_AT(a.azimuth_deg, a.tilt_deg)
        sx += a.p_kwp * nx
        sy += a.p_kwp * ny
        sz += a.p_kwp * nz
    return (sx, sy, sz)


def effective_orientation(arrays: List[SubArray]) -> Tuple[Vec3, float, float, float]:
    """
    Returns:
      n_eff   : effective unit normal (direction of N)
      P_equiv : ||N||  (equivalent kWp of a single plane with n_eff)
      A_eff   : effective azimuth (deg)
      T_eff   : effective tilt (deg)
    """
    N = capacity_vector(arrays)
    P_equiv = norm(N)
    if P_equiv == 0.0:
        raise ValueError("Resultant capacity vector is zero; check inputs.")
    n_eff = normalize(N)
    A_eff, T_eff = AT_from_unit_normal(n_eff)
    return n_eff, P_equiv, A_eff, T_eff


def effective_kwp_toward_direction(arrays: List[SubArray], s_hat: Vec3) -> float:
    """
    Exact "effective kWp" toward a given unit direction s_hat (e.g. sun direction):
        sum_i P_i * max(0, n_i dot s_hat)
    """
    s_hat = normalize(s_hat)
    total = 0.0
    for a in arrays:
        n_i = unit_normal_from_AT(a.azimuth_deg, a.tilt_deg)
        total += a.p_kwp * max(0.0, dot(n_i, s_hat))
    return total


def main() -> None:
    arrays = [
        SubArray("West",  265.0, 48.0, 3.24),
        SubArray("East",   85.0, 48.0, 4.05),
        SubArray("South", 175.0,  9.0, 2.43),
    ]

    print("Sub-arrays (Adeg, Tdeg, kWp) and unit normals n=(nx, ny, nz):")
    for a in arrays:
        nx, ny, nz = unit_normal_from_AT(a.azimuth_deg, a.tilt_deg)
        print(
            f"  - {a.name:5s}: A={a.azimuth_deg:6.1f}deg, T={a.tilt_deg:5.1f}deg, "
            f"P={a.p_kwp:4.2f} kWp  ->  n=({nx:+.6f}, {ny:+.6f}, {nz:+.6f})"
        )

    P_installed = sum(a.p_kwp for a in arrays)
    N = capacity_vector(arrays)
    n_eff, P_equiv, A_eff, T_eff = effective_orientation(arrays)

    # Persist results for reuse
    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arrays": [
            {"name": a.name, "azimuth_deg": a.azimuth_deg, "tilt_deg": a.tilt_deg, "p_kwp": a.p_kwp}
            for a in arrays
        ],
        "capacity_vector": {"x": N[0], "y": N[1], "z": N[2]},
        "effective": {
            "n_eff": {"x": n_eff[0], "y": n_eff[1], "z": n_eff[2]},
            "P_equiv_kwp": P_equiv,
            "A_eff_deg": A_eff,
            "T_eff_deg": T_eff,
        },
        "P_installed_kwp": P_installed,
    }
    OUTPUT_CONFIG.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Config written to {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()

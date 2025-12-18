"""
wing_loads.py

Computes main-wing aerodynamic + inertia loads (forces, distributions, locations)
from a saved AeroSandbox airplane (*.aero) and a saved solution JSON (soln.json).

Outputs:
- Global aero forces/moments at the reference point
- Spanwise strip loads for the main wing: location (x,y,z) and force vector (Fx,Fy,Fz)
- Battery inertia loads distributed across the inboard third (between booms)

Notes:
- VLM global outputs are documented; per-panel outputs are not guaranteed as a stable API.
  We therefore generate a conservative spanwise distribution and scale to the desired total.
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as onp

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.atmosphere import Atmosphere


g = 9.81


@dataclass
class StripLoad:
    xyz: onp.ndarray    # [m] application point in geometry axes, shape (3,)
    F_g: onp.ndarray    # [N] force in geometry axes, shape (3,)


def load_soln(soln_json_path: Path) -> dict:
    with open(soln_json_path, "r") as f:
        return json.load(f)


def get_main_wing(airplane: asb.Airplane, main_wing_name: str = "Main Wing") -> asb.Wing:
    for w in airplane.wings:
        if w.name == main_wing_name:
            return w
    raise ValueError(f"Could not find wing named '{main_wing_name}'. Available: {[w.name for w in airplane.wings]}")


def make_spanwise_stations(main_wing: asb.Wing, n_stations_per_semispan: int = 60) -> onp.ndarray:
    """
    Returns spanwise stations y from 0 -> b/2, clustered near root and tip.
    """
    b = float(main_wing.span())
    y = onp.cosspace(0.0, b / 2, n_stations_per_semispan)
    return y


def chord_at_y_linear_from_xsecs(main_wing: asb.Wing, y: onp.ndarray) -> onp.ndarray:
    """
    Approximates chord(y) by interpolating chord between WingXSecs using their y locations.
    Works well for your 3-xsec wing definition.
    """
    # Extract xsec y positions and chords (right wing)
    ys = onp.array([float(xsec.xyz_le[1]) for xsec in main_wing.xsecs])
    cs = onp.array([float(xsec.chord) for xsec in main_wing.xsecs])

    # Ensure increasing order
    order = onp.argsort(ys)
    ys = ys[order]
    cs = cs[order]

    # Interpolate chord vs y
    return onp.interp(y, ys, cs)


def z_at_y_linear_from_xsecs(main_wing: asb.Wing, y: onp.ndarray) -> onp.ndarray:
    ys = onp.array([float(xsec.xyz_le[1]) for xsec in main_wing.xsecs])
    zs = onp.array([float(xsec.xyz_le[2]) for xsec in main_wing.xsecs])
    order = onp.argsort(ys)
    ys = ys[order]
    zs = zs[order]
    return onp.interp(y, ys, zs)


def x_le_at_y_linear_from_xsecs(main_wing: asb.Wing, y: onp.ndarray) -> onp.ndarray:
    ys = onp.array([float(xsec.xyz_le[1]) for xsec in main_wing.xsecs])
    xs = onp.array([float(xsec.xyz_le[0]) for xsec in main_wing.xsecs])
    order = onp.argsort(ys)
    ys = ys[order]
    xs = xs[order]
    return onp.interp(y, ys, xs)


def elliptic_shape(y: onp.ndarray, b: float) -> onp.ndarray:
    """
    Elliptic lift shape on semispan: proportional to sqrt(1 - (2y/b)^2).
    """
    mu = 2 * y / b
    val = onp.sqrt(onp.clip(1 - mu**2, 0.0, 1.0))
    return val


def normalize_shape_to_total(shape: onp.ndarray, y: onp.ndarray, total: float) -> onp.ndarray:
    """
    Convert dimensionless shape(y) into a distributed load L'(y) such that:
    integral_0^{b/2} L'(y) dy = total/2 (semispan total).
    """
    area = onp.trapz(shape, y)
    if area <= 0:
        raise ValueError("Shape integral is zero; cannot normalize.")
    return (total / 2) * shape / area


def compute_main_wing_strip_aero_loads(
    airplane: asb.Airplane,
    operating_altitude_m: float,
    temperature_deviation_K: float,
    velocity_mps: float,
    alpha_deg: float,
    n_load_factor: float,
    W_total_N: float,
    main_wing_name: str = "Main Wing",
    n_stations_per_semispan: int = 60,
) -> tuple[dict, list[StripLoad]]:
    """
    Produces:
    - global aero dict from VLM (for bookkeeping)
    - strip loads on both wings (mirrored), in geometry axes

    Force direction convention:
    - Geometry axes in ASB: +x forward, +y right, +z up.
    - Lift is typically +z (up) in geometry for small angles; confirm sign by checking output.
    """
    atm = Atmosphere(operating_altitude_m, temperature_deviation=temperature_deviation_K)

    # Use VLM for global forces/moments (documented outputs) :contentReference[oaicite:3]{index=3}
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            atmosphere=atm,
            velocity=velocity_mps,
            alpha=alpha_deg,
            beta=0.0,
            p=0.0,
            q=0.0,
            r=0.0,
        ),
        run_symmetric_if_possible=True,
        spanwise_resolution=10,
        chordwise_resolution=10,
    )
    aero = vlm.run()  # global

    # Target total lift for this load factor:
    L_target = n_load_factor * W_total_N

    # Build a main-wing-only spanwise lift distribution.
    main_wing = get_main_wing(airplane, main_wing_name=main_wing_name)
    b = float(main_wing.span())
    y = make_spanwise_stations(main_wing, n_stations_per_semispan=n_stations_per_semispan)

    # Shape choice: elliptic (robust first-pass). Optionally replace with area-weighting.
    shape = elliptic_shape(y, b=b)
    Lprime = normalize_shape_to_total(shape, y, total=L_target)  # [N/m] (whole wing total)

    # Convert to strip forces for semispan stations
    dy = onp.diff(y, prepend=y[0])
    # Use midpoint rule-ish: local strip lift = L'(y_i) * dy_i
    dL = Lprime * dy  # semispan strip lift [N]

    # Apply at quarter-chord, on the wing reference surface
    chord = chord_at_y_linear_from_xsecs(main_wing, y)
    x_le = x_le_at_y_linear_from_xsecs(main_wing, y)
    z = z_at_y_linear_from_xsecs(main_wing, y)

    x_app = x_le + 0.25 * chord
    # Right semispan points (y >= 0) and left mirrored points (y <= 0)
    loads: list[StripLoad] = []

    for xi, yi, zi, lift_i in zip(x_app, y, z, dL):
        # Force in geometry axes: lift up (+z)
        Fg = onp.array([0.0, 0.0, float(lift_i)], dtype=float)
        loads.append(StripLoad(xyz=onp.array([float(xi), float(yi), float(zi)]), F_g=Fg))
        loads.append(StripLoad(xyz=onp.array([float(xi), float(-yi), float(zi)]), F_g=Fg))

    return aero, loads


def compute_battery_inertia_loads_inboard(
    airplane: asb.Airplane,
    battery_mass_kg: float,
    n_z: float,
    boom_spacing_frac: float,
    main_wing_name: str = "Main Wing",
    n_stations_inboard_each_side: int = 30,
) -> list[StripLoad]:
    """
    Distributes battery inertia as a vertical load over the inboard third (between booms),
    approximated as: |y| <= boom_y, where boom_y = 0.5 * boom_spacing_frac * wingspan.

    Returns downward forces (negative z) at quarter-chord.
    """
    main_wing = get_main_wing(airplane, main_wing_name=main_wing_name)
    b = float(main_wing.span())
    boom_y = 0.5 * boom_spacing_frac * b

    # Spanwise stations from centerline to boom_y
    y = onp.linspace(0.0, boom_y, n_stations_inboard_each_side)
    dy = onp.diff(y, prepend=y[0])

    # Total inertia force (vertical)
    F_total = n_z * battery_mass_kg * g  # [N] positive upward if n_z positive; we apply as inertia load downward
    # Structural convention: inertia force acts opposite acceleration; for +n_z, inertia is "down".
    # So we apply -F_total in z for +n_z case.
    Fz_total = -float(F_total)

    # Uniform distribution across the between-booms region (both sides)
    # Total span covered is 2*boom_y, so per-unit-span:
    w = Fz_total / (2 * boom_y)  # [N/m]
    dF = w * dy  # [N] on one side; mirrored makes both sides

    chord = chord_at_y_linear_from_xsecs(main_wing, y)
    x_le = x_le_at_y_linear_from_xsecs(main_wing, y)
    z = z_at_y_linear_from_xsecs(main_wing, y)
    x_app = x_le + 0.25 * chord

    loads: list[StripLoad] = []
    for xi, yi, zi, Fi in zip(x_app, y, z, dF):
        Fg = onp.array([0.0, 0.0, float(Fi)], dtype=float)
        loads.append(StripLoad(xyz=onp.array([float(xi), float(yi), float(zi)]), F_g=Fg))
        if yi != 0:
            loads.append(StripLoad(xyz=onp.array([float(xi), float(-yi), float(zi)]), F_g=Fg))

    return loads


def main():
    # ---- User inputs / paths
    run_dir = Path("output/run_31sn")  # update
    airplane_path = run_dir / "airplane.aero"    # you wrote this file
    soln_path = run_dir / "soln.json"

    airplane = asb.load(airplane_path)  # documented load() :contentReference[oaicite:4]{index=4}
    soln = load_soln(soln_path)

    # Pull what you already save
    operating_altitude_m = float(soln["Mission"]["operating_altitude"])
    temperature_deviation_K = float(soln["Environment"]["temperature_K"] - Atmosphere(operating_altitude_m).temperature())
    velocity_mps = float(soln["Performance"]["airspeed"])
    W_total_N = float(soln["Performance"]["weight_N"])

    boom_spacing_frac = float(soln["Geometry"]["boom_spacing_frac"])

    # Example load case set (adjust to your case matrix)
    alpha_deg = float(soln["Main Wing"]["struct_defined_aoa"])  # or pick a trimmed alpha if you later compute it

    # Example: +2.5g main-wing aero loads
    aero, wing_strips = compute_main_wing_strip_aero_loads(
        airplane=airplane,
        operating_altitude_m=operating_altitude_m,
        temperature_deviation_K=temperature_deviation_K,
        velocity_mps=velocity_mps,
        alpha_deg=alpha_deg,
        n_load_factor=2.5,
        W_total_N=W_total_N,
    )

    # Example: battery inertia (+2.5g vertical). You must decide battery mass extraction.
    # If your soln.json has battery mass, use it; otherwise estimate from capacity.
    battery_mass_kg = float(soln["Masses"]["mass_batteries"])
    batt_loads = compute_battery_inertia_loads_inboard(
        airplane=airplane,
        battery_mass_kg=battery_mass_kg,
        n_z=2.5,
        boom_spacing_frac=boom_spacing_frac,
    )

    # ---- Output: write a simple JSON artifact
    out = {
        "global_aero": {k: aero[k] for k in ["L", "D", "Y", "F_g", "M_g", "CL", "CD", "Cm"] if k in aero},
        "main_wing_aero_strips": [
            {"xyz_m": sl.xyz.tolist(), "F_g_N": sl.F_g.tolist()} for sl in wing_strips
        ],
        "battery_inertia_strips": [
            {"xyz_m": sl.xyz.tolist(), "F_g_N": sl.F_g.tolist()} for sl in batt_loads
        ],
    }

    out_path = run_dir / "main_wing_loads.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

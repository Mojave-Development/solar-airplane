import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as onp
from aerosandbox.weights import MassProperties
from lib.artifacts import latest_run_dir, write_json


def _mp_point(mass_kg: float, xyz: Tuple[float, float, float]) -> MassProperties:
    return MassProperties(mass=float(mass_kg), x_cg=float(xyz[0]), y_cg=float(xyz[1]), z_cg=float(xyz[2]))


def _mp_box(mass_kg: float, xyz_cg: Tuple[float, float, float], Lx: float, Ly: float, Lz: float) -> MassProperties:
    m = float(mass_kg)
    Lx = float(Lx)
    Ly = float(Ly)
    Lz = float(Lz)
    return MassProperties(
        mass=m,
        x_cg=float(xyz_cg[0]),
        y_cg=float(xyz_cg[1]),
        z_cg=float(xyz_cg[2]),
        Ixx=m / 12 * (Ly**2 + Lz**2),
        Iyy=m / 12 * (Lx**2 + Lz**2),
        Izz=m / 12 * (Lx**2 + Ly**2),
        Ixy=0.0,
        Iyz=0.0,
        Ixz=0.0,
    )


def _mp_cylinder_x(mass_kg: float, xyz_cg: Tuple[float, float, float], radius_m: float, length_m: float) -> MassProperties:
    m = float(mass_kg)
    r = float(radius_m)
    L = float(length_m)
    return MassProperties(
        mass=m,
        x_cg=float(xyz_cg[0]),
        y_cg=float(xyz_cg[1]),
        z_cg=float(xyz_cg[2]),
        Ixx=0.5 * m * r**2,
        Iyy=m / 12 * (3 * r**2 + L**2),
        Izz=m / 12 * (3 * r**2 + L**2),
        Ixy=0.0,
        Iyz=0.0,
        Ixz=0.0,
    )


def _mp_disk_x(mass_kg: float, xyz_cg: Tuple[float, float, float], radius_m: float) -> MassProperties:
    m = float(mass_kg)
    r = float(radius_m)
    return MassProperties(
        mass=m,
        x_cg=float(xyz_cg[0]),
        y_cg=float(xyz_cg[1]),
        z_cg=float(xyz_cg[2]),
        Ixx=0.5 * m * r**2,
        Iyy=0.25 * m * r**2,
        Izz=0.25 * m * r**2,
        Ixy=0.0,
        Iyz=0.0,
        Ixz=0.0,
    )


def _mp_to_dict(mp: MassProperties) -> Dict[str, float]:
    return {
        "mass": float(mp.mass),
        "x_cg": float(mp.x_cg),
        "y_cg": float(mp.y_cg),
        "z_cg": float(mp.z_cg),
        "Ixx": float(mp.Ixx),
        "Iyy": float(mp.Iyy),
        "Izz": float(mp.Izz),
        "Ixy": float(mp.Ixy),
        "Iyz": float(mp.Iyz),
        "Ixz": float(mp.Ixz),
    }


def compute_and_embed_mass_properties(soln: dict) -> dict:
    dv = soln["design_variables"]
    dg = soln["derived_geometry"]
    masses = soln["masses"]
    const = soln["constants"]
    airfoils = soln.get("airfoils", {})
    assumptions = soln.get("mass_properties_assumptions", {})

    b_w = float(dv["wingspan"])
    c_root = float(dv["chordlen"])
    boom_len = float(dv["boom_length"])
    boom_y = float(dg["boom_y"])
    x_cg_assumed = float(dg["cg_le_dist"])
    vstab_span = float(dg["vstab_span"])
    vstab_root_chord = float(dg["vstab_root_chord"])
    hstab_chord = float(dg["hstab_chordlen"])
    hstab_span = float(dg["hstab_span"])

    wing_t_over_c = float(airfoils.get("wing_t_over_c", 0.12))
    tail_t_over_c = float(airfoils.get("tail_t_over_c", 0.10))
    t_wing = wing_t_over_c * c_root
    t_tail = tail_t_over_c * max(hstab_chord, 1e-6)

    batt_box = assumptions.get("battery_box_m", [0.20, 0.10, 0.03])
    avionics_box = assumptions.get("avionics_box_m", [0.15, 0.08, 0.03])
    esc_box = assumptions.get("esc_box_m", [0.08, 0.04, 0.02])
    motor_cyl = assumptions.get("motor_cyl_m", [0.03, 0.06])
    wire_box_thickness = assumptions.get("wire_box_thickness_m", [0.01, 0.005])  # [Lx, Lz]
    pod_radius = float(assumptions.get("pod_radius_m", 0.05))
    pod_length = float(assumptions.get("pod_length_m", 0.5))

    boom_radius = float(const["boom_radius"])
    solar_panel_side_length = float(const["solar_panel_side_length"])

    wing_xyz = (x_cg_assumed, 0.0, 0.0)
    hstab_xyz = (boom_len + vstab_root_chord / 4 + 0.25 * hstab_chord, 0.0, vstab_span)

    vstab_each_mass = 0.5 * float(masses["vstab_total"])
    vstab_xyz_L = (boom_len + 0.25 * vstab_root_chord, boom_y, 0.5 * vstab_span)
    vstab_xyz_R = (boom_len + 0.25 * vstab_root_chord, -boom_y, 0.5 * vstab_span)

    boom_each_mass = 0.5 * float(masses["booms_total"])
    boom_xyz_L = (0.5 * boom_len, boom_y, -0.02)
    boom_xyz_R = (0.5 * boom_len, -boom_y, -0.02)

    pod_each_mass = 0.5 * float(masses["fuselages_total"])
    pod_xyz_L = (-0.25, boom_y, -0.02)
    pod_xyz_R = (-0.25, -boom_y, -0.02)

    solar_area = float(dv["solar_panels_n"]) * solar_panel_side_length**2
    solar_mean_chord_covered = solar_area / max(b_w, 1e-6)
    solar_xyz = (x_cg_assumed, 0.0, 0.001)

    batt_xyz = (x_cg_assumed, 0.0, 0.0)
    avionics_xyz_L = pod_xyz_L
    avionics_xyz_R = pod_xyz_R

    servo_xyz_L = (boom_len + 0.10, boom_y, vstab_span * 0.8)
    servo_xyz_R = (boom_len + 0.10, -boom_y, vstab_span * 0.8)

    prop_radius = 0.5 * float(dv["propeller_diameter"])
    motor_xyz_L = (-0.05, boom_y, 0.0)
    motor_xyz_R = (-0.05, -boom_y, 0.0)
    esc_xyz_L = (0.05, boom_y, -0.01)
    esc_xyz_R = (0.05, -boom_y, -0.01)
    prop_xyz_L = (-0.10, boom_y, 0.0)
    prop_xyz_R = (-0.10, -boom_y, 0.0)

    wire_Lx = float(wire_box_thickness[0])
    wire_Lz = float(wire_box_thickness[1])
    wire_box = (wire_Lx, max(b_w, 1e-6), wire_Lz)

    mp_components = {
        "Main wing": _mp_box(float(masses["main_wing"]), wing_xyz, Lx=0.75 * c_root, Ly=b_w, Lz=max(t_wing, 0.005)),
        "Horizontal stabilizer": _mp_box(float(masses["hstab"]), hstab_xyz, Lx=hstab_chord, Ly=hstab_span, Lz=max(t_tail, 0.003)),
        "Vertical stabilizer L": _mp_box(
            vstab_each_mass,
            vstab_xyz_L,
            Lx=max(0.5 * (vstab_root_chord + hstab_chord), 1e-6),
            Ly=max(t_tail, 0.003),
            Lz=max(vstab_span, 1e-6),
        ),
        "Vertical stabilizer R": _mp_box(
            vstab_each_mass,
            vstab_xyz_R,
            Lx=max(0.5 * (vstab_root_chord + hstab_chord), 1e-6),
            Ly=max(t_tail, 0.003),
            Lz=max(vstab_span, 1e-6),
        ),
        "Boom L": _mp_cylinder_x(boom_each_mass, boom_xyz_L, radius_m=boom_radius, length_m=boom_len),
        "Boom R": _mp_cylinder_x(boom_each_mass, boom_xyz_R, radius_m=boom_radius, length_m=boom_len),
        "Pod L": _mp_cylinder_x(pod_each_mass, pod_xyz_L, radius_m=pod_radius, length_m=pod_length),
        "Pod R": _mp_cylinder_x(pod_each_mass, pod_xyz_R, radius_m=pod_radius, length_m=pod_length),
        "Solar cells": _mp_box(float(masses["solar_cells"]), solar_xyz, Lx=max(solar_mean_chord_covered, 0.05), Ly=b_w, Lz=0.002),
        "Batteries": _mp_box(float(masses["batteries"]), batt_xyz, *batt_box),
        "Wires": _mp_box(float(masses["wires"]), wing_xyz, *wire_box),
        "Avionics L": _mp_box(0.5 * float(masses["avionics"]), avionics_xyz_L, *avionics_box),
        "Avionics R": _mp_box(0.5 * float(masses["avionics"]), avionics_xyz_R, *avionics_box),
        "Servos L": _mp_point(0.5 * float(masses["servos"]), servo_xyz_L),
        "Servos R": _mp_point(0.5 * float(masses["servos"]), servo_xyz_R),
        "Motor L": _mp_cylinder_x(0.5 * float(masses["motors_mounted"]), motor_xyz_L, radius_m=motor_cyl[0], length_m=motor_cyl[1]),
        "Motor R": _mp_cylinder_x(0.5 * float(masses["motors_mounted"]), motor_xyz_R, radius_m=motor_cyl[0], length_m=motor_cyl[1]),
        "ESC L": _mp_box(0.5 * float(masses["esc"]), esc_xyz_L, *esc_box),
        "ESC R": _mp_box(0.5 * float(masses["esc"]), esc_xyz_R, *esc_box),
        "Prop L": _mp_disk_x(0.5 * float(masses["propellers"]), prop_xyz_L, radius_m=max(prop_radius, 0.01)),
        "Prop R": _mp_disk_x(0.5 * float(masses["propellers"]), prop_xyz_R, radius_m=max(prop_radius, 0.01)),
    }

    mp_total = sum(mp_components.values(), MassProperties(mass=0.0))
    soln["mass_properties"] = {
        "components": {k: _mp_to_dict(v) for k, v in mp_components.items()},
        "total": _mp_to_dict(mp_total),
    }
    return soln


def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"
    run_dir = latest_run_dir(output_dir)
    soln_path = run_dir / "soln.json"

    with open(soln_path, "r") as f:
        soln = json.load(f)

    soln = compute_and_embed_mass_properties(soln)
    write_json(soln_path, soln, indent=2)

    print(f"[postprocess] Updated: {soln_path}")


if __name__ == "__main__":
    main()

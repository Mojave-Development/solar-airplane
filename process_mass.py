import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import aerosandbox as asb
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
    # Access data from solution structure
    main_wing = soln.get("Main Wing", {})
    geometry = soln.get("Geometry", {})
    hstab = soln.get("HStab", {})
    vstab = soln.get("V Stab", {})
    masses = soln.get("Masses", {})
    power = soln.get("Power", {})
    propulsion = soln.get("Propulsion", {})
    airfoils = soln.get("airfoils", {})

    b_w = float(main_wing.get("wingspan", main_wing.get("b_w", 0)))
    c_root = float(main_wing.get("chordlen", 0))
    boom_len = float(geometry.get("boom_length", 0))
    boom_y = float(geometry.get("boom_y", 0))
    x_cg_assumed = float(geometry.get("cg_le_dist", 0))
    vstab_span = float(vstab.get("vstab_span", 0))
    vstab_root_chord = float(vstab.get("vstab_root_chord", 0))
    hstab_chord = float(hstab.get("hstab_chordlen", 0))
    hstab_span = float(hstab.get("hstab_span", 0))

    wing_t_over_c = float(airfoils.get("wing_t_over_c", 0.12))
    tail_t_over_c = float(airfoils.get("tail_t_over_c", 0.10))
    t_wing = wing_t_over_c * c_root
    t_tail = tail_t_over_c * max(hstab_chord, 1e-6)

    boom_radius = float(geometry.get("boom_radius", 0.01))
    solar_panel_side_length = float(power.get("solar_panel_side_length", 0.125))
    
    # Fuselage dimensions from opti.py
    # Fuselage length: 0.5m (from x=-0.5 to x=0)
    # Fuselage radius: 0.4 * dae51 airfoil max thickness
    fuselage_length = 0.5
    fuselage_radius = 0.4 * asb.Airfoil("dae51").max_thickness()
    
    # Battery box dimensions for moment of inertia calculation
    batt_box = [0.20, 0.10, 0.03]  # [Lx, Ly, Lz] in meters

    wing_xyz = (x_cg_assumed, 0.0, 0.0)
    hstab_xyz = (boom_len + vstab_root_chord / 4 + 0.25 * hstab_chord, 0.0, vstab_span)

    # Vstab masses (split between left and right)
    vstab_total_mass = float(masses.get("mass_vstab", 0))
    vstab_each_mass = 0.5 * vstab_total_mass
    vstab_xyz_L = (boom_len + 0.25 * vstab_root_chord, boom_y, 0.5 * vstab_span)
    vstab_xyz_R = (boom_len + 0.25 * vstab_root_chord, -boom_y, 0.5 * vstab_span)

    # Boom masses (split between left and right)
    boom_total_mass = float(masses.get("mass_boom", 0))
    boom_each_mass = 0.5 * boom_total_mass
    boom_xyz_L = (0.5 * boom_len, boom_y, -0.02)
    boom_xyz_R = (0.5 * boom_len, -boom_y, -0.02)

    # Fuselage masses (split between left and right)
    fuselage_total_mass = float(masses.get("mass_fuselages", 0))
    fuselage_each_mass = 0.5 * fuselage_total_mass
    fuselage_xyz_L = (-0.25, boom_y, -0.02)  # Middle of fuselage
    fuselage_xyz_R = (-0.25, -boom_y, -0.02)

    # Solar cells - distributed across full span starting from center
    solar_panels_n = float(power.get("solar_panels_n", 0))
    solar_area = solar_panels_n * solar_panel_side_length**2
    solar_mean_chord_covered = solar_area / max(b_w, 1e-6)
    solar_xyz = (x_cg_assumed + 0.25 * c_root, 0.0, 0.001)  # Quarter chord, centerline

    # Batteries - distributed across inboard wing (y=-boom_y to y=+boom_y)
    batt_xyz = (x_cg_assumed + 0.25 * c_root, 0.0, 0.0)  # Quarter chord, centerline

    # Avionics components at middle of main wing at third chord
    avionics_x = x_cg_assumed + 0.33 * c_root  # Third chord
    avionics_xyz = (avionics_x, 0.0, 0.0)  # Centerline

    # Propulsion components
    prop_radius = 0.5 * float(propulsion.get("propeller_diameter", 0.4))
    # Motors at front of fuselages
    motor_xyz_L = (-0.5, boom_y, -0.02)  # Front of fuselage
    motor_xyz_R = (-0.5, -boom_y, -0.02)
    # ESC at middle of fuselages
    esc_xyz_L = (-0.25, boom_y, -0.01)  # Middle of fuselage
    esc_xyz_R = (-0.25, -boom_y, -0.01)
    # Propellers at front of fuselage
    prop_xyz_L = (-0.5, boom_y, 0.0)  # Front of fuselage
    prop_xyz_R = (-0.5, -boom_y, 0.0)

    # Extract individual avionics component masses
    mass_fc = float(masses.get("mass_fc", 0.08))
    mass_gps = float(masses.get("mass_gps", 0.02))
    mass_telemtry = float(masses.get("mass_telemtry", 0.03))
    mass_receiver = float(masses.get("mass_receiver", 0.02))
    # Navlights: 4 total, 0.01 kg each
    mass_navlights_total = float(masses.get("mass_navlights", 0.04))
    mass_navlight_each = mass_navlights_total / 4.0
    
    # Power board mass (located at superstructure position)
    mass_power_board = float(masses.get("mass_power_board", 0.15))

    # Navlight positions
    navlight_port_xyz = (avionics_x, -b_w / 4, 0.0)  # Port wing
    navlight_starboard_xyz = (avionics_x, b_w / 4, 0.0)  # Starboard wing
    navlight_center_xyz = (avionics_x, 0.0, 0.0)  # Center with avionics
    navlight_hstab_xyz = (hstab_xyz[0] + 0.33 * hstab_chord, 0.0, hstab_xyz[2])  # Hstab center

    # Structural interface positions
    # Superstructures: middle of wing where booms are
    superstructure_each_mass = 0.5 * float(masses.get("mass_superstructures", 0))
    superstructure_xyz_L = (x_cg_assumed, boom_y, 0.0)
    superstructure_xyz_R = (x_cg_assumed, -boom_y, 0.0)
    
    # Power board: located at superstructure position (middle of wing where booms are)
    power_board_each_mass = 0.5 * mass_power_board
    power_board_xyz_L = (x_cg_assumed, boom_y, 0.0)
    power_board_xyz_R = (x_cg_assumed, -boom_y, 0.0)

    # Boom_vstab interfaces: ends of each boom
    boom_vstab_each_mass = 0.5 * float(masses.get("mass_boom_vstab_interfaces", 0))
    boom_vstab_xyz_L = (boom_len, boom_y, -0.02)
    boom_vstab_xyz_R = (boom_len, -boom_y, -0.02)

    # Vstab & hstab interfaces: where vstab and hstab meet
    vstab_hstab_each_mass = 0.5 * float(masses.get("mass_vstab_stab_interfaces", 0))
    vstab_hstab_xyz_L = (boom_len + vstab_root_chord / 4, boom_y, vstab_span)
    vstab_hstab_xyz_R = (boom_len + vstab_root_chord / 4, -boom_y, vstab_span)

    # Build mass properties components
    mp_components = {
        "Main wing": _mp_box(float(masses.get("mass_main_wing", 0)), wing_xyz, Lx=0.75 * c_root, Ly=b_w, Lz=max(t_wing, 0.005)),
        "Horizontal stabilizer": _mp_box(float(masses.get("mass_hstab", 0)), hstab_xyz, Lx=hstab_chord, Ly=hstab_span, Lz=max(t_tail, 0.003)),
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
        "Fuselage L": _mp_cylinder_x(fuselage_each_mass, fuselage_xyz_L, radius_m=fuselage_radius, length_m=fuselage_length),
        "Fuselage R": _mp_cylinder_x(fuselage_each_mass, fuselage_xyz_R, radius_m=fuselage_radius, length_m=fuselage_length),
        # Solar cells: distributed across full span (Ly = wingspan)
        "Solar cells": _mp_box(float(masses.get("mass_solar_cells", 0)), solar_xyz, Lx=max(solar_mean_chord_covered, 0.05), Ly=b_w, Lz=0.002),
        # Batteries: distributed across inboard wing (Ly = 2 * boom_y)
        "Batteries": _mp_box(float(masses.get("mass_batteries", 0)), batt_xyz, Lx=batt_box[0], Ly=2 * boom_y, Lz=batt_box[2]),
        # Wires: Disregard (removed)
        # Avionics components at third chord
        "FC": _mp_point(mass_fc, avionics_xyz),
        "GPS": _mp_point(mass_gps, avionics_xyz),
        "Telemetry": _mp_point(mass_telemtry, avionics_xyz),
        "Receiver": _mp_point(mass_receiver, avionics_xyz),
        # Navlights
        "Navlight Port": _mp_point(mass_navlight_each, navlight_port_xyz),
        "Navlight Starboard": _mp_point(mass_navlight_each, navlight_starboard_xyz),
        "Navlight Center": _mp_point(mass_navlight_each, navlight_center_xyz),
        "Navlight Hstab": _mp_point(mass_navlight_each, navlight_hstab_xyz),
        "Motor L": _mp_point(0.5 * float(masses.get("mass_motors_mounted", 0)), motor_xyz_L),
        "Motor R": _mp_point(0.5 * float(masses.get("mass_motors_mounted", 0)), motor_xyz_R),
        "ESC L": _mp_point(0.5 * float(masses.get("mass_esc", 0)), esc_xyz_L),
        "ESC R": _mp_point(0.5 * float(masses.get("mass_esc", 0)), esc_xyz_R),
        "Prop L": _mp_disk_x(0.5 * float(masses.get("mass_propellers", 0)), prop_xyz_L, radius_m=max(prop_radius, 0.01)),
        "Prop R": _mp_disk_x(0.5 * float(masses.get("mass_propellers", 0)), prop_xyz_R, radius_m=max(prop_radius, 0.01)),
        # Structural interfaces
        "Superstructure L": _mp_point(superstructure_each_mass, superstructure_xyz_L),
        "Superstructure R": _mp_point(superstructure_each_mass, superstructure_xyz_R),
        "Power board L": _mp_point(power_board_each_mass, power_board_xyz_L),
        "Power board R": _mp_point(power_board_each_mass, power_board_xyz_R),
        "Boom_vstab interface L": _mp_point(boom_vstab_each_mass, boom_vstab_xyz_L),
        "Boom_vstab interface R": _mp_point(boom_vstab_each_mass, boom_vstab_xyz_R),
        "Vstab_hstab interface L": _mp_point(vstab_hstab_each_mass, vstab_hstab_xyz_L),
        "Vstab_hstab interface R": _mp_point(vstab_hstab_each_mass, vstab_hstab_xyz_R),
    }

    mp_total = sum(mp_components.values(), MassProperties(mass=0.0))
    
    # Verify total mass matches solution
    total_mass_calculated = mp_total.mass
    total_mass_expected = float(masses.get("total_mass", 0))
    if abs(total_mass_calculated - total_mass_expected) > 0.01:  # 10g tolerance
        print(f"[WARNING] Total mass mismatch: calculated={total_mass_calculated:.6f} kg, expected={total_mass_expected:.6f} kg")
    
    mass_properties = {
        "components": {k: _mp_to_dict(v) for k, v in mp_components.items()},
        "total": _mp_to_dict(mp_total),
        "total_mass_verification": {
            "calculated": total_mass_calculated,
            "expected": total_mass_expected,
            "difference": total_mass_calculated - total_mass_expected,
        }
    }
    return mass_properties


def main(soln_path: Path):
    with open(soln_path, "r") as f:
        soln = json.load(f)

    mass_properties = compute_and_embed_mass_properties(soln)
    
    # Write mass properties to a new JSON file
    mass_properties_path = soln_path.parent / "mass_properties.json"
    write_json(mass_properties_path, mass_properties, indent=2)

    print(f"[postprocess] Mass properties written to: {mass_properties_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
        soln_path = run_dir / "soln.json"
        if not soln_path.exists():
            raise FileNotFoundError(f"Solution file not found: {soln_path}")
    else:
        base_dir = Path(__file__).resolve().parent
        output_dir = base_dir / "output"
        run_dir = latest_run_dir(output_dir)
        soln_path = run_dir / "soln.json"
    
    main(soln_path)

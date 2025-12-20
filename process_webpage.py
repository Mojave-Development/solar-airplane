#!/usr/bin/env python3
"""
Generate a standalone Three.js-based aircraft mass viewer HTML file.

Key features:
- Embeds soln.json + mass_properties.json directly into a single HTML file
- Renders basic 3D shapes for components (boxes, cylinders, spheres)
- Optional wireframe from an AeroSandbox .aero airplane file
- Sidebar aircraft specs with tabs
- Legend toggles component visibility
- Hover tooltip with mass + inertia info
- Optional Chart.js battery/power plot if battery_states are present
- Energy balance chart FIXED: embeds energy_balance_data.csv into HTML (no runtime fetch)
"""

from __future__ import annotations

import csv
import html
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import aerosandbox as asb

from lib.artifacts import latest_run_dir

JSONDict = Dict[str, Any]


# ----------------------------
# Small HTML helpers
# ----------------------------

def _h(text: str) -> str:
    """HTML-escape."""
    return html.escape(text, quote=True)


def _num(v: Any, default: float = 0.0) -> float:
    """Best-effort numeric conversion."""
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _row(label: str, value: str) -> str:
    return f"<tr><td><b>{_h(label)}:</b></td><td>{value}</td></tr>"


def _fmt(v: Any, fmt: str, default: float = 0.0, suffix: str = "") -> str:
    """Format numeric value with fallback."""
    return f"{_num(v, default):{fmt}}{suffix}"


def _section(title: str, rows: Iterable[str]) -> str:
    rows_html = "".join(rows)
    return (
        f"<h3>{_h(title)}</h3>"
        "<table style='width: 100%; border-collapse: collapse;'>"
        f"{rows_html}"
        "</table>"
    )


# ----------------------------
# Energy balance CSV -> embedded JSON
# ----------------------------

def load_energy_balance_csv(run_dir: Path) -> Optional[JSONDict]:
    """
    Load energy balance CSV and return series data for plotting.

    Expected columns (by index; matches your prior JS parsing):
      time_hr              -> col 0
      power_generated_W    -> col 3
      power_used_W         -> col 4
      battery_state_Wh     -> col 6

    If your CSV changes, update indices or switch to header-based parsing.
    """
    candidates = [
        run_dir / "energy_balance_data.csv",
        run_dir / "analysis" / "energy_balance_data.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return None

    text = csv_path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 2:
        return None

    time_hr: List[float] = []
    period_type: List[str] = []
    p_gen: List[float] = []
    p_used: List[float] = []
    batt_Wh: List[float] = []

    for r in rows[1:]:  # skip header
        if not r or len(r) < 7:
            continue
        try:
            t = float(r[0])
            period = str(r[1]) if len(r) > 1 else "day"
            g = float(r[3])
            u = float(r[4])
            b = float(r[6])
        except (ValueError, IndexError):
            continue

        time_hr.append(t)
        period_type.append(period)
        p_gen.append(g)
        p_used.append(u)
        batt_Wh.append(b)

    if not time_hr:
        return None

    if csv_path.parent == run_dir:
        source_csv = csv_path.name
    else:
        source_csv = f"{csv_path.parent.name}/{csv_path.name}"

    return {
        "source_csv": source_csv,
        "time_hr": time_hr,
        "period_type": period_type,
        "power_generated_W": p_gen,
        "power_used_W": p_used,
        "battery_state_Wh": batt_Wh,
    }


# ----------------------------
# Specs HTML
# ----------------------------

def format_specs_html(soln: Mapping[str, Any], mass_properties: Mapping[str, Any]) -> str:
    """Format aircraft specifications as HTML."""
    meta = soln.get("Meta", {}) or {}
    perf = soln.get("Performance", {}) or {}
    main_wing = soln.get("Main Wing", {}) or {}
    geom = soln.get("Geometry", {}) or {}
    total = (mass_properties.get("total", {}) or {})
    aero = soln.get("Aerodynamics", {}) or {}
    mission = soln.get("Mission", {}) or {}
    env = soln.get("Environment", {}) or {}
    hstab = soln.get("HStab", {}) or {}
    vstab = soln.get("V Stab", {}) or soln.get("V_Stab", {}) or {}
    power = soln.get("Power", {}) or {}
    prop = soln.get("Propulsion", {}) or {}
    masses = soln.get("Masses", {}) or {}

    tab_contents: Dict[str, List[str]] = {
        "overview": [],
        "performance": [],
        "aerodynamics": [],
        "wings": [],
        "geometry": [],
        "mass": [],
        "power": [],
        "propulsion": [],
        "structure": [],
    }

    # Overview tab
    if meta:
        meta_rows = [_row("Run ID", _h(str(meta.get("run_id", "N/A"))))]
        if meta.get("timestamp_utc"):
            meta_rows.append(_row("Timestamp (UTC)", _h(str(meta.get("timestamp_utc")))))
        tab_contents["overview"].append(_section("Meta", meta_rows))

    if mission:
        tab_contents["overview"].append(_section("Mission", [
            _row("Mission Date", _h(str(mission.get("mission_date", "N/A")))),
            _row("Operating Latitude", _fmt(mission.get("operating_lat"), ".4f", suffix="°")),
            _row("Operating Altitude", _fmt(mission.get("operating_altitude"), ".0f", suffix=" m")),
        ]))

    if env:
        tab_contents["overview"].append(_section("Environment", [
            _row("Temperature (K)", _fmt(env.get("temperature_K"), ".2f", suffix=" K")),
            _row("Temperature (F)", _fmt(env.get("temperature_F"), ".1f", suffix=" °F")),
            _row("Pressure", _fmt(env.get("pressure"), ".0f", suffix=" Pa")),
            _row("Density", _fmt(env.get("density"), ".4f", suffix=" kg/m³")),
            _row("Speed of Sound", _fmt(env.get("speed_of_sound"), ".2f", suffix=" m/s")),
        ]))

    # Performance tab
    tab_contents["performance"].append(_section("Performance", [
        _row("Total Mass", _fmt(perf.get("total_mass"), ".3f", suffix=" kg")),
        _row("Airspeed", _fmt(perf.get("airspeed"), ".2f", suffix=" m/s")),
        _row("Thrust (Cruise)", _fmt(perf.get("thrust_cruise"), ".2f", suffix=" N")),
        _row("Thrust (Climb)", _fmt(perf.get("thrust_climb"), ".2f", suffix=" N")),
        _row("Power (Cruise, All Motors)", _fmt(perf.get("power_cruise (all motors)"), ".2f", suffix=" W")),
        _row("Power (Cruise, One Motor)", _fmt(perf.get("power_cruise (one motor)"), ".2f", suffix=" W")),
        _row("Power (Shaft Cruise, One Motor)", _fmt(perf.get("power_shaft_cruise (one motor)"), ".2f", suffix=" W")),
        _row("Power Out Max", _fmt(perf.get("power_out_max"), ".2f", suffix=" W")),
        _row("Min Climb Angle", _fmt(perf.get("min_climb_angle"), ".1f", suffix="°")),
        _row("L/D Ratio", _fmt(perf.get("L_over_D"), ".2f")),
        _row("Stall Speed", _fmt(perf.get("stall_speed"), ".2f", suffix=" m/s")),
        _row("Takeoff Gross Weight", _fmt(perf.get("togw"), ".2f", suffix=" N")),
    ]))

    # Aerodynamics tab
    tab_contents["aerodynamics"].append(_section("Aerodynamics", [
        _row("CL", _fmt(aero.get("CL"), ".3f")),
        _row("CD", _fmt(aero.get("CD"), ".4f")),
        _row("Cm", _fmt(aero.get("Cm"), ".4f")),
        _row("Lift", _fmt(aero.get("L"), ".2f", suffix=" N")),
        _row("Drag", _fmt(aero.get("D"), ".2f", suffix=" N")),
        _row("Static Margin", _fmt(aero.get("static_margin"), ".3f")),
        _row("Neutral Point X", _fmt(aero.get("x_np"), ".3f", suffix=" m")),
    ]))

    # Wings tab
    main_wing_rows = [
        _row("Wingspan", _fmt(main_wing.get("wingspan"), ".3f", suffix=" m")),
        _row("Chord", _fmt(main_wing.get("chordlen"), ".3f", suffix=" m")),
        _row("Area", _fmt(main_wing.get("S_w"), ".3f", suffix=" m²")),
        _row("Aspect Ratio", _fmt(main_wing.get("main_wing_AR"), ".2f")),
    ]
    if main_wing.get("struct_defined_aoa") is not None:
        main_wing_rows.append(_row("Structural AOA", _fmt(main_wing.get("struct_defined_aoa"), ".2f", suffix="°")))
    if main_wing.get("MAC_w") is not None:
        main_wing_rows.append(_row("Mean Aerodynamic Chord", _fmt(main_wing.get("MAC_w"), ".3f", suffix=" m")))
    if main_wing.get("wing_airfoil"):
        main_wing_rows.append(_row("Airfoil", _h(str(main_wing.get("wing_airfoil", "N/A")))))
    tab_contents["wings"].append(_section("Main Wing", main_wing_rows))

    if hstab:
        tab_contents["wings"].append(_section("Horizontal Stabilizer", [
            _row("Span", _fmt(hstab.get("hstab_span"), ".3f", suffix=" m")),
            _row("Chord", _fmt(hstab.get("hstab_chordlen"), ".3f", suffix=" m")),
            _row("Area", _fmt(hstab.get("hstab_area"), ".3f", suffix=" m²")),
            _row("Aspect Ratio", _fmt(hstab.get("hstab_AR"), ".2f")),
            _row("Angle of Attack", _fmt(hstab.get("hstab_aoa"), ".2f", suffix="°")),
            _row("Volume Coefficient", _fmt(hstab.get("V_H_actual"), ".3f")),
            _row("Airfoil", _h(str(hstab.get("hstab_airfoil", "N/A")))),
        ]))

    if vstab:
        tab_contents["wings"].append(_section("Vertical Stabilizer", [
            _row("Span (Height)", _fmt(vstab.get("vstab_span"), ".3f", suffix=" m")),
            _row("Root Chord", _fmt(vstab.get("vstab_root_chord"), ".3f", suffix=" m")),
            _row("Area (each)", _fmt(vstab.get("vstab_area"), ".3f", suffix=" m²")),
            _row("Total Area", _fmt(vstab.get("vstab_area_total"), ".3f", suffix=" m²")),
            _row("Aspect Ratio", _fmt(vstab.get("vstab_AR"), ".2f")),
            _row("Volume Coefficient", _fmt(vstab.get("V_V_actual"), ".4f")),
            _row("Airfoil", _h(str(vstab.get("vstab_airfoil", "N/A")))),
        ]))

    # Geometry tab
    tab_contents["geometry"].append(_section("Geometry", [
        _row("Boom Length", _fmt(geom.get("boom_length"), ".3f", suffix=" m")),
        _row("Boom Y Position", _fmt(geom.get("boom_y"), ".3f", suffix=" m")),
        _row("CG Location (from LE)", _fmt(geom.get("cg_le_dist"), ".3f", suffix=" m")),
    ]))

    # Mass tab
    tab_contents["mass"].append(_section("Mass Properties", [
        _row("Total Mass", _fmt(total.get("mass"), ".3f", suffix=" kg")),
        _row("CG X", _fmt(total.get("x_cg"), ".3f", suffix=" m")),
        _row("CG Y", _fmt(total.get("y_cg"), ".3f", suffix=" m")),
        _row("CG Z", _fmt(total.get("z_cg"), ".3f", suffix=" m")),
        _row("Ixx", _fmt(total.get("Ixx"), ".3f", suffix=" kg·m²")),
        _row("Iyy", _fmt(total.get("Iyy"), ".3f", suffix=" kg·m²")),
        _row("Izz", _fmt(total.get("Izz"), ".3f", suffix=" kg·m²")),
    ]))

    if masses:
        masses_rows = [
            _row("Solar Cells", _fmt(masses.get("mass_solar_cells"), ".3f", suffix=" kg")),
            _row("Power Board", _fmt(masses.get("mass_power_board"), ".3f", suffix=" kg")),
            _row("Batteries", _fmt(masses.get("mass_batteries"), ".3f", suffix=" kg")),
            _row("Wires", _fmt(masses.get("mass_wires"), ".3f", suffix=" kg")),
            _row("Avionics", _fmt(masses.get("mass_avionics"), ".3f", suffix=" kg")),
            _row("Servos", _fmt(masses.get("mass_servos"), ".3f", suffix=" kg")),
            _row("Motors (Mounted)", _fmt(masses.get("mass_motors_mounted"), ".3f", suffix=" kg")),
            _row("ESCs", _fmt(masses.get("mass_escs"), ".3f", suffix=" kg")),
            _row("Propellers", _fmt(masses.get("mass_propellers"), ".3f", suffix=" kg")),
            _row("Main Wing", _fmt(masses.get("mass_main_wing"), ".3f", suffix=" kg")),
            _row("Horizontal Stabilizer", _fmt(masses.get("mass_hstab"), ".3f", suffix=" kg")),
            _row("Vertical Stabilizer", _fmt(masses.get("mass_vstab"), ".3f", suffix=" kg")),
            _row("Boom", _fmt(masses.get("mass_boom"), ".3f", suffix=" kg")),
            _row("Fuselages", _fmt(masses.get("mass_fuselages"), ".3f", suffix=" kg")),
            _row("Superstructures", _fmt(masses.get("mass_superstructures"), ".3f", suffix=" kg")),
            _row("Boom-VStab Interfaces", _fmt(masses.get("mass_boom_vstab_interfaces"), ".3f", suffix=" kg")),
            _row("VStab-HStab Interfaces", _fmt(masses.get("mass_vstab_stab_interfaces"), ".3f", suffix=" kg")),
            _row("Total Mass", _fmt(masses.get("total_mass"), ".3f", suffix=" kg")),
        ]
        tab_contents["mass"].append(_section("Masses Breakdown", masses_rows))

    # Power tab
    if power:
        power_rows = [
            _row("Solar Panels", _fmt(power.get("solar_panels_n"), ".1f")),
            _row("Solar Panels (Rows)", _fmt(power.get("solar_panels_n_rows"), ".0f")),
            _row("Battery Capacity", _fmt(power.get("battery_capacity"), ".1f", suffix=" Wh")),
            _row("Battery Voltage", _fmt(power.get("battery_voltage"), ".1f", suffix=" V")),
            _row("Number of Packs", _fmt(power.get("num_packs"), ".1f")),
        ]
        if power.get("solar_panel_side_length") is not None:
            power_rows.append(_row("Panel Size", _fmt(_num(power.get("solar_panel_side_length")) * 1000, ".1f", suffix=" mm")))
        if power.get("solar_encapsulation_eff_hit") is not None:
            power_rows.append(_row("Solar Encapsulation Efficiency (HIT)", _fmt(power.get("solar_encapsulation_eff_hit"), ".3f")))
        if power.get("solar_cell_efficiency") is not None:
            power_rows.append(_row("Solar Cell Efficiency", _fmt(power.get("solar_cell_efficiency"), ".3f")))
        if power.get("energy_generation_margin") is not None:
            power_rows.append(_row("Energy Generation Margin", _fmt(power.get("energy_generation_margin"), ".3f")))
        tab_contents["power"].append(_section("Power System", power_rows))

        battery_states = power.get("battery_states") or []
        if isinstance(battery_states, list) and len(battery_states) > 0:
            tab_contents["power"].append(
                "<div style='margin-top: 15px;'>"
                "<h4 style='margin-bottom: 10px; color: #4CAF50;'>Power Generation & Battery State</h4>"
                "<div style='position: relative; height: 200px; width: 100%; max-height: 200px; overflow: hidden;'>"
                "<canvas id='powerChart' style='max-height: 200px;'></canvas>"
                "</div>"
                "</div>"
            )

    # Energy balance container (always include; JS will show message if missing)
    tab_contents["power"].append(
        "<div style='margin-top: 20px; padding-top: 15px; border-top: 2px solid #444;'>"
        "<h4 style='margin-bottom: 10px; color: #4CAF50;'>Energy Balance Analysis</h4>"
        "<div style='position: relative; height: 300px; width: 100%; max-height: 300px; overflow: hidden;'>"
        "<canvas id='energyBalanceChart' style='max-height: 300px;'></canvas>"
        "</div>"
        "</div>"
    )

    # Propulsion tab
    if prop:
        prop_rows: List[str] = []
        if prop.get("propeller_diameter") is not None:
            prop_rows.append(_row("Propeller Diameter", _fmt(prop.get("propeller_diameter"), ".3f", suffix=" m")))
        if prop.get("propeller_n") is not None:
            prop_rows.append(_row("Number of Motors", _fmt(prop.get("propeller_n"), ".0f")))
        if prop.get("advanced_ratio") is not None:
            prop_rows.append(_row("Advanced Ratio", _fmt(prop.get("advanced_ratio"), ".3f")))
        if prop.get("motor_kv") is not None:
            prop_rows.append(_row("Motor KV", _fmt(prop.get("motor_kv"), ".2f", suffix=" rpm/V")))
        if prop.get("rpm_cruise") is not None:
            prop_rows.append(_row("RPM (Cruise)", _fmt(prop.get("rpm_cruise"), ".1f", suffix=" rpm")))
        if prop.get("i_cruise") is not None:
            prop_rows.append(_row("Current (Cruise)", _fmt(prop.get("i_cruise"), ".3f", suffix=" A")))
        if prop_rows:
            tab_contents["propulsion"].append(_section("Propulsion", prop_rows))

    # Structure tab
    if geom.get("boom_radius") is not None:
        tab_contents["structure"].append(_section("Structural Details", [
            _row("Boom Radius", _fmt(_num(geom.get("boom_radius")) * 1000, ".1f", suffix=" mm")),
            _row("Boom Spacing Fraction", _fmt(geom.get("boom_spacing_frac"), ".2f")),
        ]))

    # Build tab HTML structure
    tab_names = {
        "overview": "Overview",
        "performance": "Performance",
        "aerodynamics": "Aerodynamics",
        "wings": "Wings",
        "geometry": "Geometry",
        "mass": "Mass",
        "power": "Power",
        "propulsion": "Propulsion",
        "structure": "Structure",
    }

    parts: List[str] = []
    parts.append("<div style='font-family: Arial, sans-serif; padding: 0; margin-bottom: 40px;'>")
    parts.append("<h2 style='margin-top: 0;'>Aircraft Specifications</h2>")

    parts.append("<div class='tabs-container'>")
    parts.append("<div class='tabs'>")
    for tab_id, tab_name in tab_names.items():
        active_class = " active" if tab_id == "overview" else ""
        parts.append(f"<button class='tab-button{active_class}' data-tab='{tab_id}'>{_h(tab_name)}</button>")
    parts.append("</div>")

    for tab_id in tab_names:
        active_class = " active" if tab_id == "overview" else ""
        parts.append(f"<div class='tab-content{active_class}' id='tab-{tab_id}'>")
        parts.extend(tab_contents[tab_id])
        parts.append("</div>")

    parts.append("</div>")  # tabs-container
    parts.append("</div>")  # wrapper
    return "".join(parts)


# ----------------------------
# AeroSandbox geometry extraction
# ----------------------------

def extract_airplane_geometry(airplane_path: Path) -> JSONDict:
    """Extract geometry from an AeroSandbox .aero airplane file for wireframe visualization."""
    if not airplane_path.exists():
        return {}

    try:
        airplane = asb.load(str(airplane_path))
    except Exception as e:
        print(f"[warning] Could not load airplane file {airplane_path}: {e}")
        return {}

    geometry: JSONDict = {"wings": [], "fuselages": []}

    for wing in getattr(airplane, "wings", []) or []:
        wing_data: JSONDict = {
            "name": getattr(wing, "name", "Unknown"),
            "symmetric": bool(getattr(wing, "symmetric", False)),
            "xsecs": [],
        }
        for xsec in getattr(wing, "xsecs", []) or []:
            xyz_le = getattr(xsec, "xyz_le", [0, 0, 0])
            if hasattr(xyz_le, "tolist"):
                xyz_le = xyz_le.tolist()
            wing_data["xsecs"].append({
                "xyz_le": list(xyz_le),
                "chord": float(getattr(xsec, "chord", 0.0) or 0.0),
            })
        geometry["wings"].append(wing_data)

    for fuselage in getattr(airplane, "fuselages", []) or []:
        fuselage_data: JSONDict = {"name": getattr(fuselage, "name", "Unknown"), "xsecs": []}
        for xsec in getattr(fuselage, "xsecs", []) or []:
            xyz_c = getattr(xsec, "xyz_c", [0, 0, 0])
            if hasattr(xyz_c, "tolist"):
                xyz_c = xyz_c.tolist()
            fuselage_data["xsecs"].append({
                "xyz_c": list(xyz_c),
                "radius": float(getattr(xsec, "radius", 0.0) or 0.0),
            })
        geometry["fuselages"].append(fuselage_data)

    xyz_ref = getattr(airplane, "xyz_ref", [0, 0, 0])
    if hasattr(xyz_ref, "tolist"):
        xyz_ref = xyz_ref.tolist()
    geometry["xyz_ref"] = list(xyz_ref)
    return geometry


# ----------------------------
# HTML template
# ----------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Aircraft Mass Viewer - Three.js</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: Arial, sans-serif; overflow-x: hidden; overflow-y: auto; background: #1a1a1a; color: #e0e0e0; }}

    #canvas-container {{
      width: 100vw;
      height: 60vh;
      position: relative;
      background: #1a1a1a;
    }}

    #content-area {{
      padding: 0;
      width: 100%;
      margin: 0;
    }}

    .tabs-container {{
      margin-top: 20px;
    }}

    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-bottom: 20px;
      border-bottom: 2px solid #444;
    }}

    .tab-button {{
      background: rgba(40,40,40,0.8);
      color: #b0b0b0;
      border: none;
      padding: 10px 20px;
      cursor: pointer;
      font-size: 14px;
      border-radius: 5px 5px 0 0;
      transition: all 0.3s ease;
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
    }}

    .tab-button:hover {{
      background: rgba(50,50,50,0.9);
      color: #e0e0e0;
    }}

    .tab-button.active {{
      background: rgba(30,30,30,0.95);
      color: #4CAF50;
      border-bottom: 2px solid #4CAF50;
    }}

    .tab-content {{
      display: none;
    }}

    .tab-content.active {{
      display: block;
    }}

    #legend {{
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(30,30,30,0.95);
      padding: 15px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.5);
      min-width: 200px;
      max-width: 250px;
      max-height: calc(60vh - 20px);
      overflow-y: auto;
      z-index: 1000;
      border: 1px solid #444;
    }}
    #legend h3 {{ margin-top: 0; margin-bottom: 10px; color: #4CAF50; font-size: 16px; }}
    .legend-item {{ display: flex; align-items: center; padding: 5px 0; cursor: pointer; user-select: none; color: #e0e0e0; }}
    .legend-item:hover {{ background: rgba(255,255,255,0.1); border-radius: 4px; }}
    .legend-color {{ width: 20px; height: 20px; border: 1px solid #555; margin-right: 8px; flex-shrink: 0; border-radius: 3px; }}
    .legend-item.hidden {{ opacity: 0.3; }}

    #sidebar {{
      flex: 1;
      background: rgba(30,30,30,0.95);
      padding: 20px;
      border-radius: 0px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.5);
      min-width: 400px;
    }}
    #sidebar h1 {{ margin-top: 0; color: #4CAF50; font-size: 24px; }}
    #sidebar h2 {{ margin-top: 0; color: #4CAF50; font-size: 20px; }}
    #sidebar h3 {{ margin-top: 15px; margin-bottom: 10px; color: #4CAF50; font-size: 16px; }}
    #sidebar h4 {{ margin-top: 10px; margin-bottom: 8px; color: #4CAF50; font-size: 14px; }}
    #sidebar p {{ color: #aaa; }}
    #sidebar table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }}
    #sidebar table td {{ padding: 8px; border-bottom: 1px solid #444; color: #e0e0e0; }}
    #sidebar table td:first-child {{ font-weight: bold; width: 60%; color: #b0b0b0; }}

    #info {{
      position: fixed;
      bottom: 10px;
      left: 10px;
      background: rgba(0,0,0,0.8);
      color: #e0e0e0;
      padding: 10px 15px;
      border-radius: 5px;
      font-size: 12px;
      z-index: 1000;
      border: 1px solid #444;
    }}

    #tooltip {{
      position: fixed;
      background: rgba(20,20,20,0.95);
      color: #e0e0e0;
      padding: 10px 15px;
      border-radius: 5px;
      font-size: 12px;
      pointer-events: none;
      z-index: 2000;
      display: none;
      max-width: 320px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.5);
      border: 1px solid #444;
    }}
    #tooltip.visible {{ display: block; }}
    #tooltip h4 {{ margin: 0 0 5px 0; color: #4CAF50; font-size: 14px; }}
    #tooltip p {{ margin: 3px 0; color: #e0e0e0; }}
  </style>
</head>
<body>
  <div id="canvas-container">
    <div id="legend">
      <h3>Components</h3>
      <div id="legend-content"></div>
    </div>
  </div>

  <div id="content-area">
    <div id="sidebar">
      {specs_html}
    </div>
  </div>

  <div id="info">
    Left click: Rotate | Right click: Pan | Scroll: Zoom
  </div>

  <div id="tooltip"></div>

  <script type="importmap">
    {{
      "imports": {{
        "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
      }}
    }}
  </script>

  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

    const soln = {soln_json};
    const massProperties = {mass_properties_json};
    const airplaneGeometry = {airplane_geometry_json};
    const energyBalanceData = {energy_balance_data_json};

    const batteryStates = soln.Power?.battery_states || null;
    const batteryCapacity = soln.Power?.battery_capacity || 0;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Align to expected aircraft view: rotate scene instead of remapping every component
    scene.rotation.z = Math.PI;
    scene.rotation.x = -Math.PI / 2;

    const canvasContainer = document.getElementById('canvas-container');
    const canvasHeight = Math.floor(window.innerHeight * 0.6);
    const canvasWidth = window.innerWidth;

    const camera = new THREE.PerspectiveCamera(75, canvasWidth / canvasHeight, 0.1, 1000);
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(canvasWidth, canvasHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    canvasContainer.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    function getColorForCategory(name) {{
      const n = (name || '').toLowerCase();
      if (n.includes('wing') || n.includes('stabilizer')) return 0x4A90E2;
      if (n.includes('boom') || n.includes('fuselage') || n.includes('superstructure') || n.includes('interface')) return 0x8B7355;
      if (n.includes('motor') || n.includes('esc') || n.includes('prop')) return 0xE67E22;
      if (n.includes('batter')) return 0xE74C3C;
      if (n.includes('solar')) return 0xF1C40F;
      if (n.includes('fc') || n.includes('gps') || n.includes('telemetry') || n.includes('receiver') || n.includes('navlight') || n.includes('power board')) return 0x27AE60;
      return 0x9B59B6;
    }}

    // Lightweight materials + geometry (reduced segments)
    function createMaterial(color, opacity=0.7) {{
      return new THREE.MeshPhongMaterial({{
        color,
        opacity,
        transparent: opacity < 1.0,
        side: THREE.DoubleSide,
        emissive: 0x000000,
        emissiveIntensity: 0.0
      }});
    }}

    function createBox(center, Lx, Ly, Lz, color, userData) {{
      const geometry = new THREE.BoxGeometry(Lx, Ly, Lz); // no subdivisions
      const material = createMaterial(color, 0.7);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(center[0], center[1], center[2]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = userData;
      return mesh;
    }}

    function createCylinder(center, radius, length, axis, color, userData) {{
      const geometry = new THREE.CylinderGeometry(radius, radius, length, 24);
      const material = createMaterial(color, 0.7);
      const mesh = new THREE.Mesh(geometry, material);

      if (axis === 'x') mesh.rotation.z = Math.PI / 2;
      else if (axis === 'z') mesh.rotation.x = Math.PI / 2;

      mesh.position.set(center[0], center[1], center[2]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = userData;
      return mesh;
    }}

    function createSphere(center, radius, color, userData) {{
      const geometry = new THREE.SphereGeometry(radius, 16, 12);
      const material = createMaterial(color, 0.8);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(center[0], center[1], center[2]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = userData;
      return mesh;
    }}

    function reconstructGeometry(soln, massProperties) {{
      const mainWing = soln['Main Wing'] || soln.Main_Wing || {{}};
      const geometry = soln.Geometry || {{}};
      const hstab = soln.HStab || {{}};
      const vstab = soln['V Stab'] || soln.V_Stab || soln.VStab || {{}};
      const power = soln.Power || {{}};
      const propulsion = soln.Propulsion || {{}};
      const airfoils = soln.airfoils || {{}};

      const b_w = mainWing.wingspan || mainWing.b_w || 0;
      const c_root = mainWing.chordlen || 0;
      const boom_len = geometry.boom_length || 0;
      const boom_y = geometry.boom_y || 0;
      const vstab_span = vstab.vstab_span || 0;
      const vstab_root_chord = vstab.vstab_root_chord || 0;
      const hstab_chord = hstab.hstab_chordlen || 0;
      const hstab_span = hstab.hstab_span || 0;

      const wing_t_over_c = airfoils.wing_t_over_c || 0.12;
      const tail_t_over_c = airfoils.tail_t_over_c || 0.10;
      const t_wing = wing_t_over_c * c_root;
      const t_tail = tail_t_over_c * Math.max(hstab_chord, 1e-6);

      const boom_radius = geometry.boom_radius || 0.01;
      const solar_panel_side_length = power.solar_panel_side_length || 0.125;

      const fuselage_length = 0.5;
      const fuselage_radius = 0.015;
      const batt_box = [0.20, 0.10, 0.03];

      const solar_panels_n = power.solar_panels_n || 0;
      const solar_area = solar_panels_n * solar_panel_side_length * solar_panel_side_length;
      const solar_mean_chord_covered = solar_area / Math.max(b_w, 1e-6);

      const prop_radius = 0.5 * (propulsion.propeller_diameter || 0.4);

      const components = massProperties.components || {{}};
      const geometryMap = {{}};

      const masses = Object.values(components).map(c => c.mass || 0);
      const maxMass = Math.max(1e-9, ...(masses.length ? masses : [1e-9]));

      for (const [name, comp] of Object.entries(components)) {{
        const x_cg = comp.x_cg || 0;
        const y_cg = comp.y_cg || 0;
        const z_cg = comp.z_cg || 0;
        const mass = comp.mass || 0;

        if (name === 'Main wing') {{
          geometryMap[name] = {{ type: 'box', center: [x_cg, y_cg, z_cg], Lx: 0.75 * c_root, Ly: b_w, Lz: Math.max(t_wing, 0.005), mass }};
        }} else if (name === 'Horizontal stabilizer') {{
          geometryMap[name] = {{ type: 'box', center: [x_cg, y_cg, z_cg], Lx: hstab_chord, Ly: hstab_span, Lz: Math.max(t_tail, 0.003), mass }};
        }} else if (name.includes('Vertical stabilizer')) {{
          geometryMap[name] = {{ type: 'box', center: [x_cg, y_cg, z_cg], Lx: Math.max(0.5 * (vstab_root_chord + hstab_chord), 1e-6), Ly: Math.max(t_tail, 0.003), Lz: Math.max(vstab_span, 1e-6), mass }};
        }} else if (name === 'Boom L' || name === 'Boom R') {{
          geometryMap[name] = {{ type: 'cylinder', center: [x_cg, y_cg, z_cg], radius: boom_radius, length: boom_len, axis: 'x', mass }};
        }} else if (name === 'Fuselage L' || name === 'Fuselage R') {{
          geometryMap[name] = {{ type: 'cylinder', center: [x_cg, y_cg, z_cg], radius: fuselage_radius, length: fuselage_length, axis: 'x', mass }};
        }} else if (name === 'Solar cells') {{
          geometryMap[name] = {{ type: 'box', center: [x_cg, y_cg, z_cg], Lx: Math.max(solar_mean_chord_covered, 0.05), Ly: b_w, Lz: 0.002, mass }};
        }} else if (name === 'Batteries') {{
          geometryMap[name] = {{ type: 'box', center: [x_cg, y_cg, z_cg], Lx: batt_box[0], Ly: 2 * boom_y, Lz: batt_box[2], mass }};
        }} else if (name === 'Prop L' || name === 'Prop R') {{
          geometryMap[name] = {{ type: 'cylinder', center: [x_cg, y_cg, z_cg], radius: Math.max(prop_radius, 0.01), length: 0.001, axis: 'x', mass }};
        }} else {{
          const sphereRadius = 0.05 * (mass / maxMass) + 0.02;
          geometryMap[name] = {{ type: 'point', center: [x_cg, y_cg, z_cg], radius: sphereRadius, mass }};
        }}
      }}

      return geometryMap;
    }}

    const geometryMap = reconstructGeometry(soln, massProperties);
    const components = massProperties.components || {{}};

    const componentMeshes = {{}};
    const legendContent = document.getElementById('legend-content');

    for (const [name, geom] of Object.entries(geometryMap)) {{
      const color = getColorForCategory(name);
      const componentData = components[name] || {{}};

      const inertia = (componentData.Ixx !== undefined) ? {{
        Ixx: componentData.Ixx, Iyy: componentData.Iyy, Izz: componentData.Izz,
        Ixy: componentData.Ixy || 0, Iyz: componentData.Iyz || 0, Ixz: componentData.Ixz || 0
      }} : null;

      const userData = {{
        name, type: geom.type, mass: geom.mass, center: geom.center, inertia
      }};

      let mesh = null;
      if (geom.type === 'box') mesh = createBox(geom.center, geom.Lx, geom.Ly, geom.Lz, color, userData);
      else if (geom.type === 'cylinder') mesh = createCylinder(geom.center, geom.radius, geom.length, geom.axis, color, userData);
      else if (geom.type === 'point') mesh = createSphere(geom.center, geom.radius, color, userData);

      if (!mesh) continue;

      scene.add(mesh);
      componentMeshes[name] = mesh;

      const legendItem = document.createElement('div');
      legendItem.className = 'legend-item';
      legendItem.innerHTML = `
        <div class="legend-color" style="background-color: #${{color.toString(16).padStart(6, '0')}}"></div>
        <span>${{name}}</span>
      `;
      legendItem.onclick = () => {{
        mesh.visible = !mesh.visible;
        legendItem.classList.toggle('hidden', !mesh.visible);
      }};
      legendContent.appendChild(legendItem);
    }}

    // Total CG marker
    const total = massProperties.total || {{}};
    const cgGeometry = new THREE.SphereGeometry(0.05, 16, 12);
    const cgMaterial = new THREE.MeshPhongMaterial({{ color: 0xff0000, emissive: 0x000000, emissiveIntensity: 0.0 }});
    const cgMesh = new THREE.Mesh(cgGeometry, cgMaterial);
    cgMesh.position.set(total.x_cg || 0, total.y_cg || 0, total.z_cg || 0);
    cgMesh.userData = {{
      name: 'Total CG', type: 'cg', mass: total.mass || 0,
      center: [total.x_cg || 0, total.y_cg || 0, total.z_cg || 0],
      inertia: null
    }};
    scene.add(cgMesh);
    componentMeshes['Total CG'] = cgMesh;

    // Wireframe tube helper
    function createWireframeTube(start, end, radius=0.003, color=0x808080) {{
      const direction = new THREE.Vector3().subVectors(end, start);
      const length = direction.length();
      const geometry = new THREE.CylinderGeometry(radius, radius, length, 8);
      const material = new THREE.MeshBasicMaterial({{ color, depthWrite: false }});
      const mesh = new THREE.Mesh(geometry, material);

      const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      mesh.position.copy(midpoint);

      const axis = new THREE.Vector3(0, 1, 0);
      const quaternion = new THREE.Quaternion();
      quaternion.setFromUnitVectors(axis, direction.normalize());
      mesh.quaternion.copy(quaternion);

      mesh.renderOrder = 1000;
      return mesh;
    }}

    function addWireframe(airplaneGeometry) {{
      if (!airplaneGeometry?.wings?.length) return;

      const wireframeColor = 0x808080;
      const wireframeRadius = 0.003;
      const xyz_ref = airplaneGeometry.xyz_ref || [0,0,0];
      const cgX = (massProperties.total || {{}}).x_cg || 0;

      for (const wing of airplaneGeometry.wings) {{
        const xsecs = wing.xsecs || [];
        if (xsecs.length < 2) continue;

        function le(xsec) {{
          return new THREE.Vector3(xsec.xyz_le[0] + xyz_ref[0] - cgX, xsec.xyz_le[1] + xyz_ref[1], xsec.xyz_le[2] + xyz_ref[2]);
        }}
        function te(xsec) {{
          return new THREE.Vector3(xsec.xyz_le[0] + xsec.chord + xyz_ref[0] - cgX, xsec.xyz_le[1] + xyz_ref[1], xsec.xyz_le[2] + xyz_ref[2]);
        }}

        for (let i = 0; i < xsecs.length - 1; i++) {{
          const a = xsecs[i], b = xsecs[i+1];
          scene.add(createWireframeTube(le(a), le(b), wireframeRadius, wireframeColor));
          scene.add(createWireframeTube(te(a), te(b), wireframeRadius, wireframeColor));
          scene.add(createWireframeTube(le(a), te(a), wireframeRadius, wireframeColor));
        }}
        scene.add(createWireframeTube(le(xsecs[xsecs.length-1]), te(xsecs[xsecs.length-1]), wireframeRadius, wireframeColor));

        if (wing.symmetric) {{
          for (let i = 0; i < xsecs.length - 1; i++) {{
            const a = xsecs[i], b = xsecs[i+1];
            const leA = le(a); leA.y = -leA.y + 2*xyz_ref[1];
            const leB = le(b); leB.y = -leB.y + 2*xyz_ref[1];
            const teA = te(a); teA.y = -teA.y + 2*xyz_ref[1];
            const teB = te(b); teB.y = -teB.y + 2*xyz_ref[1];

            scene.add(createWireframeTube(leA, leB, wireframeRadius, wireframeColor));
            scene.add(createWireframeTube(teA, teB, wireframeRadius, wireframeColor));
            scene.add(createWireframeTube(leA, teA, wireframeRadius, wireframeColor));
          }}
          const last = xsecs[xsecs.length-1];
          const leL = le(last); leL.y = -leL.y + 2*xyz_ref[1];
          const teL = te(last); teL.y = -teL.y + 2*xyz_ref[1];
          scene.add(createWireframeTube(leL, teL, wireframeRadius, wireframeColor));
        }}
      }}

      for (const fuselage of airplaneGeometry.fuselages || []) {{
        const xsecs = fuselage.xsecs || [];
        if (xsecs.length < 2) continue;
        for (let i = 0; i < xsecs.length - 1; i++) {{
          const a = xsecs[i], b = xsecs[i+1];
          const c1 = new THREE.Vector3(a.xyz_c[0] + xyz_ref[0] + cgX, a.xyz_c[1] + xyz_ref[1], a.xyz_c[2] + xyz_ref[2]);
          const c2 = new THREE.Vector3(b.xyz_c[0] + xyz_ref[0] - cgX, b.xyz_c[1] + xyz_ref[1], b.xyz_c[2] + xyz_ref[2]);
          scene.add(createWireframeTube(c1, c2, wireframeRadius, wireframeColor));
        }}
      }}
    }}

    addWireframe(airplaneGeometry);

    scene.add(new THREE.AxesHelper(2));
    const gridHelper = new THREE.GridHelper(10, 10);
    gridHelper.rotation.x = -Math.PI / 2;
    scene.add(gridHelper);

    function createAxisLabel(text, position, color) {{
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = 64; canvas.height = 64;
      ctx.fillStyle = color;
      ctx.font = 'Bold 48px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, 32, 32);

      const texture = new THREE.CanvasTexture(canvas);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({{ map: texture }}));
      sprite.position.set(position[0], position[1], position[2]);
      sprite.scale.set(0.5, 0.5, 1);
      return sprite;
    }}

    scene.add(createAxisLabel('X', [2.5, 0, 0], '#ff0000'));
    scene.add(createAxisLabel('Y', [0, 2.5, 0], '#00ff00'));
    scene.add(createAxisLabel('Z', [0, 0, 2.5], '#0000ff'));

    // Hover tooltip + highlight (no material cloning; adjust emissive)
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const tooltip = document.getElementById('tooltip');
    const allMeshes = Object.values(componentMeshes);

    let hovered = null;

    function setHover(mesh, enabled) {{
      if (!mesh || !mesh.material) return;
      if (!mesh.material.emissive) return;
      mesh.material.emissive.setHex(enabled ? 0xffffff : 0x000000);
      mesh.material.emissiveIntensity = enabled ? 0.35 : 0.0;
      mesh.material.needsUpdate = true;
    }}

    function onMouseMove(event) {{
      const canvasRect = canvasContainer.getBoundingClientRect();
      mouse.x = ((event.clientX - canvasRect.left) / canvasRect.width) * 2 - 1;
      mouse.y = -((event.clientY - canvasRect.top) / canvasRect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(allMeshes, true);

      if (hits.length) {{
        const obj = hits[0].object;
        if (hovered !== obj) {{
          if (hovered) setHover(hovered, false);
          hovered = obj;
          setHover(obj, true);

          const ud = obj.userData || {{}};
          tooltip.style.left = (event.clientX + 10) + 'px';
          tooltip.style.top = (event.clientY + 10) + 'px';

          let content = `<h4>${{ud.name || 'Component'}}</h4>`;
          content += `<p><b>Type:</b> ${{ud.type}}</p>`;
          content += `<p><b>Mass:</b> ${{(ud.mass ?? 0).toFixed(4)}} kg</p>`;
          if (ud.center) {{
            content += `<p><b>Position:</b> (${{ud.center[0].toFixed(3)}}, ${{ud.center[1].toFixed(3)}}, ${{ud.center[2].toFixed(3)}}) m</p>`;
          }}
          if (ud.inertia) {{
            content += `<p><b>Inertia (kg·m²):</b></p>`;
            content += `<p style="margin-left: 10px;">Ixx: ${{ud.inertia.Ixx.toFixed(6)}}</p>`;
            content += `<p style="margin-left: 10px;">Iyy: ${{ud.inertia.Iyy.toFixed(6)}}</p>`;
            content += `<p style="margin-left: 10px;">Izz: ${{ud.inertia.Izz.toFixed(6)}}</p>`;
            if (ud.inertia.Ixy || ud.inertia.Iyz || ud.inertia.Ixz) {{
              content += `<p style="margin-left: 10px;">Ixy: ${{(ud.inertia.Ixy || 0).toFixed(6)}}</p>`;
              content += `<p style="margin-left: 10px;">Iyz: ${{(ud.inertia.Iyz || 0).toFixed(6)}}</p>`;
              content += `<p style="margin-left: 10px;">Ixz: ${{(ud.inertia.Ixz || 0).toFixed(6)}}</p>`;
            }}
          }}

          tooltip.innerHTML = content;
          tooltip.classList.add('visible');
        }} else {{
          tooltip.style.left = (event.clientX + 10) + 'px';
          tooltip.style.top = (event.clientY + 10) + 'px';
        }}
      }} else {{
        if (hovered) {{
          setHover(hovered, false);
          hovered = null;
        }}
        tooltip.classList.remove('visible');
      }}
    }}

    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('mouseleave', () => {{
      if (hovered) setHover(hovered, false);
      hovered = null;
      tooltip.classList.remove('visible');
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}

    window.addEventListener('resize', () => {{
      const h = Math.floor(window.innerHeight * 0.6);
      const w = window.innerWidth;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    }});

    animate();

    // Battery/power chart
    if (batteryStates && batteryStates.length > 0) {{
      const canvas = document.getElementById('powerChart');
      if (canvas) {{
        canvas.style.height = '200px';
        canvas.style.maxHeight = '200px';

        const timeHours = Array.from({{length: batteryStates.length}}, (_, i) => (i / batteryStates.length) * 24);
        const dt = 24 / batteryStates.length;

        const powerGeneration = [];
        for (let i = 0; i < batteryStates.length - 1; i++) {{
          const dE = batteryStates[i + 1] - batteryStates[i];
          powerGeneration.push(dE / dt);
        }}
        if (powerGeneration.length) powerGeneration.push(powerGeneration[powerGeneration.length - 1]);

        new Chart(canvas, {{
          type: 'line',
          data: {{
            labels: timeHours.map(t => t.toFixed(1)),
            datasets: [
              {{
                label: 'Battery State (Wh)',
                data: batteryStates,
                borderColor: 'rgb(76, 175, 80)',
                backgroundColor: 'rgba(76, 175, 80, 0.2)',
                yAxisID: 'y',
                tension: 0.1,
                pointRadius: 0
              }},
              {{
                label: 'Power Generation (W)',
                data: powerGeneration,
                borderColor: 'rgb(255, 152, 0)',
                backgroundColor: 'rgba(255, 152, 0, 0.2)',
                yAxisID: 'y1',
                tension: 0.1,
                pointRadius: 0
              }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
              title: {{ display: true, text: '24-Hour Power Generation & Battery State', color: '#4CAF50' }},
              legend: {{ display: true, position: 'top', labels: {{ color: '#e0e0e0' }} }}
            }},
            scales: {{
              x: {{
                display: true,
                title: {{ display: true, text: 'Time (hours)', color: '#e0e0e0' }},
                ticks: {{ color: '#aaa' }},
                grid: {{ color: '#444' }}
              }},
              y: {{
                type: 'linear',
                display: true,
                position: 'left',
                title: {{ display: true, text: 'Battery State (Wh)', color: '#e0e0e0' }},
                ticks: {{ color: '#aaa' }},
                grid: {{ color: '#444' }},
                min: 0,
                max: batteryCapacity ? batteryCapacity * 1.1 : undefined
              }},
              y1: {{
                type: 'linear',
                display: true,
                position: 'right',
                title: {{ display: true, text: 'Power Generation (W)', color: '#e0e0e0' }},
                ticks: {{ color: '#aaa' }},
                grid: {{ drawOnChartArea: false, color: '#444' }}
              }}
            }}
          }}
        }});
      }}
    }}

    // Energy balance chart (embedded data; no fetch)
    let energyBalanceChart = null;

    function initEnergyBalanceChart() {{
      if (energyBalanceChart) return;

      const canvas = document.getElementById('energyBalanceChart');
      if (!canvas) return;

      const tabContent = canvas.closest('.tab-content');
      if (!tabContent || !tabContent.classList.contains('active')) return;

      if (!energyBalanceData) {{
        const parent = canvas.parentElement;
        parent.innerHTML = "<div style='color:#aaa; font-size:13px; padding:10px; border:1px dashed #555; border-radius:6px;'>No energy balance data found (energy_balance_data.csv).</div>";
        return;
      }}

      const timeHours = energyBalanceData.time_hr || [];
      const periodType = energyBalanceData.period_type || [];
      const powerGenerated = energyBalanceData.power_generated_W || [];
      const powerUsed = energyBalanceData.power_used_W || [];
      const batteryState = energyBalanceData.battery_state_Wh || [];
      if (!timeHours.length) return;

      canvas.style.height = '300px';
      canvas.style.maxHeight = '300px';

      // Calculate battery state range for y axis with padding
      const validBatteryState = batteryState.filter(v => !isNaN(v) && isFinite(v));
      const battMin = validBatteryState.length > 0 ? Math.min(...validBatteryState) : 0;
      const battMax = validBatteryState.length > 0 ? Math.max(...validBatteryState) : 100;
      const battRange = battMax - battMin;
      const battPadding = Math.max(battRange * 0.1, 5); // 10% padding or at least 5Wh
      const battYMin = Math.max(0, battMin - battPadding);
      const battYMax = battMax + battPadding;

      // Calculate power range for y1 axis with padding
      const allPowerValues = [...powerGenerated, ...powerUsed].filter(v => !isNaN(v) && isFinite(v));
      const powerMin = allPowerValues.length > 0 ? Math.min(...allPowerValues) : 0;
      const powerMax = allPowerValues.length > 0 ? Math.max(...allPowerValues) : 100;
      const powerRange = powerMax - powerMin;
      const powerPadding = Math.max(powerRange * 0.1, 10); // 10% padding or at least 10W
      const powerYMin = Math.min(0, powerMin - powerPadding);
      const powerYMax = powerMax + powerPadding;

      // Create background datasets for period highlighting
      // Use the maximum range that covers both y and y1 axes for full coverage
      const fullYMin = Math.min(battYMin, powerYMin);
      const fullYMax = Math.max(battYMax, powerYMax);
      
      // Create data arrays for period backgrounds (fill from min to max)
      const createPeriodBackground = (periodName, color) => {{
        const data = timeHours.map((t, i) => {{
          const period = periodType[i] || 'day';
          // Return fullYMax where period matches, fullYMin otherwise to create fill effect
          return (period === periodName || (periodName === 'dawn_dusk' && (period === 'dawn_dusk' || period === 'dusk'))) ? fullYMax : fullYMin;
        }});
        return {{
          label: periodName === 'dawn_dusk' ? 'Dawn/Dusk' : periodName.charAt(0).toUpperCase() + periodName.slice(1),
          data: data,
          backgroundColor: color,
          borderColor: 'transparent',
          borderWidth: 0,
          fill: {{ value: fullYMin }},
          pointRadius: 0,
          pointHoverRadius: 0,
          order: -1, // Render behind main datasets
          yAxisID: 'y',
          tension: 0
        }};
      }};

      const backgroundDatasets = [];
      if (periodType.length > 0) {{
        // Night background (dark blue)
        backgroundDatasets.push(createPeriodBackground('night', 'rgba(0, 0, 128, 0.20)'));
        // Dawn/Dusk background (orange)
        backgroundDatasets.push(createPeriodBackground('dawn_dusk', 'rgba(255, 165, 0, 0.18)'));
        // Day background (light yellow) - subtle indication
        backgroundDatasets.push(createPeriodBackground('day', 'rgba(255, 250, 205, 0.15)'));
      }}

      energyBalanceChart = new Chart(canvas, {{
        type: 'line',
        data: {{
          labels: timeHours.map(t => t.toFixed(2)),
          datasets: [
            ...backgroundDatasets,
            {{
              label: 'Power Generated (W)',
              data: powerGenerated,
              borderColor: 'rgb(76, 175, 80)',
              backgroundColor: 'rgba(76, 175, 80, 0.2)',
              yAxisID: 'y1',
              tension: 0.1,
              pointRadius: 0
            }},
            {{
              label: 'Power Used (W)',
              data: powerUsed,
              borderColor: 'rgb(244, 67, 54)',
              backgroundColor: 'rgba(244, 67, 54, 0.2)',
              yAxisID: 'y1',
              tension: 0.1,
              pointRadius: 0
            }},
            {{
              label: 'Battery State (Wh)',
              data: batteryState,
              borderColor: 'rgb(33, 150, 243)',
              backgroundColor: 'rgba(33, 150, 243, 0.2)',
              yAxisID: 'y',
              tension: 0.1,
              pointRadius: 0
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          plugins: {{
            title: {{
              display: true,
              text: `Energy Balance Analysis${{energyBalanceData.source_csv ? " (" + energyBalanceData.source_csv + ")" : ""}}`,
              color: '#4CAF50'
            }},
            legend: {{ display: true, position: 'top', labels: {{ color: '#e0e0e0' }} }}
          }},
          scales: {{
            x: {{
              display: true,
              title: {{ display: true, text: 'Time (hours)', color: '#e0e0e0' }},
              ticks: {{ color: '#aaa' }},
              grid: {{ color: '#444' }}
            }},
            y: {{
              type: 'linear',
              display: true,
              position: 'left',
              title: {{ display: true, text: 'Battery State (Wh)', color: '#e0e0e0' }},
              ticks: {{ color: '#aaa' }},
              grid: {{ color: '#444' }},
              min: battYMin,
              max: battYMax
            }},
            y1: {{
              type: 'linear',
              display: true,
              position: 'right',
              title: {{ display: true, text: 'Power (W)', color: '#e0e0e0' }},
              ticks: {{ color: '#aaa' }},
              grid: {{ drawOnChartArea: false, color: '#444' }},
              min: powerYMin,
              max: powerYMax
            }}
          }}
        }}
      }});
    }}

    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {{
      button.addEventListener('click', () => {{
        const tabId = button.getAttribute('data-tab');

        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        button.classList.add('active');
        const tabContent = document.getElementById(`tab-${{tabId}}`);
        if (tabContent) {{
          tabContent.classList.add('active');
          if (tabId === 'power') {{
            // allow layout to settle before Chart.js init
            setTimeout(() => initEnergyBalanceChart(), 50);
          }}
        }}
      }});
    }});
  </script>
</body>
</html>
"""


# ----------------------------
# Viewer generator
# ----------------------------

def create_threejs_viewer(
    soln_path: Path,
    mass_properties_path: Path,
    airplane_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Create a Three.js-based viewer HTML file. Returns the output HTML path."""
    if output_path is None:
        output_path = soln_path.parent / "aircraft_mass_viewer.html"

    soln = json.loads(soln_path.read_text(encoding="utf-8"))
    mass_properties = json.loads(mass_properties_path.read_text(encoding="utf-8"))

    run_dir = soln_path.parent
    specs_html = format_specs_html(soln, mass_properties)

    airplane_geometry: JSONDict = {}
    if airplane_path and airplane_path.suffix.lower() == ".aero" and airplane_path.exists():
        airplane_geometry = extract_airplane_geometry(airplane_path)

    energy_balance_data = load_energy_balance_csv(run_dir)

    # Compact JSON to keep HTML size reasonable
    soln_json = json.dumps(soln, separators=(",", ":"))
    mass_properties_json = json.dumps(mass_properties, separators=(",", ":"))
    airplane_geometry_json = json.dumps(airplane_geometry, separators=(",", ":"))
    energy_balance_data_json = json.dumps(energy_balance_data, separators=(",", ":")) if energy_balance_data else "null"

    html_content = HTML_TEMPLATE.format(
        specs_html=specs_html,
        soln_json=soln_json,
        mass_properties_json=mass_properties_json,
        airplane_geometry_json=airplane_geometry_json,
        energy_balance_data_json=energy_balance_data_json,
    )

    output_path.write_text(html_content, encoding="utf-8")
    print(f"[viewer] Saved Three.js visualization: {output_path}")
    return output_path


# ----------------------------
# CLI
# ----------------------------

def resolve_inputs(arg: Optional[str]) -> Tuple[Path, Path, Optional[Path], Path]:
    """
    Resolve file paths from CLI argument.
    - If arg is a directory: use that directory as run_dir.
    - If arg is a name: interpret as output/<name> under this script folder.
    - If arg is omitted: use latest run dir under output/.
    """
    base_dir = Path(__file__).resolve().parent
    output_dir_base = base_dir / "output"

    if arg is None:
        run_dir = latest_run_dir(output_dir_base)
    else:
        p = Path(arg)
        if p.exists() and p.is_dir():
            run_dir = p.resolve()
        else:
            run_dir = (output_dir_base / arg).resolve()

    soln_path = run_dir / "soln.json"
    mass_properties_path = run_dir / "mass_properties.json"

    airplane_path = run_dir / "airplane.aero"
    if not airplane_path.exists():
        airplane_path = None

    return soln_path, mass_properties_path, airplane_path, run_dir


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    soln_path, mass_properties_path, airplane_path, run_dir = resolve_inputs(arg)

    if not soln_path.exists():
        raise FileNotFoundError(f"Solution file not found: {soln_path}")
    if not mass_properties_path.exists():
        raise FileNotFoundError(f"Mass properties file not found: {mass_properties_path}")

    print(f"[viewer] Loading solution from: {soln_path}")
    print(f"[viewer] Loading mass properties from: {mass_properties_path}")
    if airplane_path:
        print(f"[viewer] Loading airplane geometry from: {airplane_path}")

    out_html = create_threejs_viewer(
        soln_path,
        mass_properties_path,
        airplane_path,
        run_dir / "aircraft_mass_viewer.html",
    )
    print(f"[viewer] Complete! Open {out_html} in a browser.")


if __name__ == "__main__":
    main()

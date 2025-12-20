#!/usr/bin/env python3
"""
Generate a standalone Three.js-based aircraft mass viewer HTML file.

Features:
- Loads soln.json and mass_properties.json and embeds them into a standalone HTML
- Renders basic 3D shapes for components (boxes, cylinders, spheres)
- Optional wireframe from an AeroSandbox .aero airplane file
- Sidebar aircraft specs
- Legend toggles component visibility
- Hover tooltip with mass + inertia info
- Optional Chart.js battery/power plot if battery_states are present
"""

from __future__ import annotations

import html
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
# Specs HTML
# ----------------------------

def format_specs_html(soln: Mapping[str, Any], mass_properties: Mapping[str, Any]) -> str:
    """Format aircraft specifications as HTML."""
    perf = soln.get("Performance", {}) or {}
    main_wing = soln.get("Main Wing", {}) or {}
    geom = soln.get("Geometry", {}) or {}
    total = (mass_properties.get("total", {}) or {})
    aero = soln.get("Aerodynamics", {}) or {}
    mission = soln.get("Mission", {}) or {}
    env = soln.get("Environment", {}) or {}
    hstab = soln.get("HStab", {}) or {}
    vstab = soln.get("V Stab", {}) or soln.get("V Stab", {}) or {}
    power = soln.get("Power", {}) or {}
    prop = soln.get("Propulsion", {}) or {}

    parts: List[str] = []
    parts.append("<div style='font-family: Arial, sans-serif; padding: 10px;'>")
    parts.append("<h2 style='margin-top: 0;'>Aircraft Specifications</h2>")

    parts.append(_section("Performance", [
        _row("Total Mass", _fmt(perf.get("total_mass"), ".3f", suffix=" kg")),
        _row("Airspeed", _fmt(perf.get("airspeed"), ".2f", suffix=" m/s")),
        _row("Thrust (Cruise)", _fmt(perf.get("thrust_cruise"), ".2f", suffix=" N")),
        _row("Power (Cruise)", _fmt(perf.get("power_cruise (all motors)"), ".2f", suffix=" W")),
        _row("L/D Ratio", _fmt(perf.get("L_over_D"), ".2f")),
        _row("Stall Speed", _fmt(perf.get("stall_speed"), ".2f", suffix=" m/s")),
    ]))

    parts.append(_section("Main Wing", [
        _row("Wingspan", _fmt(main_wing.get("wingspan"), ".3f", suffix=" m")),
        _row("Chord", _fmt(main_wing.get("chordlen"), ".3f", suffix=" m")),
        _row("Area", _fmt(main_wing.get("S_w"), ".3f", suffix=" m²")),
        _row("Aspect Ratio", _fmt(main_wing.get("main_wing_AR"), ".2f")),
    ]))

    parts.append(_section("Geometry", [
        _row("Boom Length", _fmt(geom.get("boom_length"), ".3f", suffix=" m")),
        _row("Boom Y Position", _fmt(geom.get("boom_y"), ".3f", suffix=" m")),
        _row("CG Location (from LE)", _fmt(geom.get("cg_le_dist"), ".3f", suffix=" m")),
    ]))

    parts.append(_section("Mass Properties", [
        _row("Total Mass", _fmt(total.get("mass"), ".3f", suffix=" kg")),
        _row("CG X", _fmt(total.get("x_cg"), ".3f", suffix=" m")),
        _row("CG Y", _fmt(total.get("y_cg"), ".3f", suffix=" m")),
        _row("CG Z", _fmt(total.get("z_cg"), ".3f", suffix=" m")),
        _row("Ixx", _fmt(total.get("Ixx"), ".3f", suffix=" kg·m²")),
        _row("Iyy", _fmt(total.get("Iyy"), ".3f", suffix=" kg·m²")),
        _row("Izz", _fmt(total.get("Izz"), ".3f", suffix=" kg·m²")),
    ]))

    parts.append(_section("Aerodynamics", [
        _row("CL", _fmt(aero.get("CL"), ".3f")),
        _row("CD", _fmt(aero.get("CD"), ".4f")),
        _row("Lift", _fmt(aero.get("L"), ".2f", suffix=" N")),
        _row("Drag", _fmt(aero.get("D"), ".2f", suffix=" N")),
        _row("Static Margin", _fmt(aero.get("static_margin"), ".3f")),
        _row("Neutral Point X", _fmt(aero.get("x_np"), ".3f", suffix=" m")),
    ]))

    if mission:
        parts.append(_section("Mission", [
            _row("Mission Date", _h(str(mission.get("mission_date", "N/A")))),
            _row("Operating Latitude", _fmt(mission.get("operating_lat"), ".4f", suffix="°")),
            _row("Operating Altitude", _fmt(mission.get("operating_altitude"), ".0f", suffix=" m")),
        ]))

    if env:
        parts.append(_section("Environment", [
            _row("Temperature", _fmt(env.get("temperature_F"), ".1f", suffix=" °F")),
            _row("Pressure", _fmt(env.get("pressure"), ".0f", suffix=" Pa")),
            _row("Density", _fmt(env.get("density"), ".4f", suffix=" kg/m³")),
        ]))

    if hstab:
        parts.append(_section("Horizontal Stabilizer", [
            _row("Span", _fmt(hstab.get("hstab_span"), ".3f", suffix=" m")),
            _row("Chord", _fmt(hstab.get("hstab_chordlen"), ".3f", suffix=" m")),
            _row("Area", _fmt(hstab.get("hstab_area"), ".3f", suffix=" m²")),
            _row("Aspect Ratio", _fmt(hstab.get("hstab_AR"), ".2f")),
            _row("Angle of Attack", _fmt(hstab.get("hstab_aoa"), ".2f", suffix="°")),
            _row("Volume Coefficient", _fmt(hstab.get("V_H_actual"), ".3f")),
            _row("Airfoil", _h(str(hstab.get("hstab_airfoil", "N/A")))),
        ]))

    if vstab:
        parts.append(_section("Vertical Stabilizer", [
            _row("Span (Height)", _fmt(vstab.get("vstab_span"), ".3f", suffix=" m")),
            _row("Root Chord", _fmt(vstab.get("vstab_root_chord"), ".3f", suffix=" m")),
            _row("Area (each)", _fmt(vstab.get("vstab_area"), ".3f", suffix=" m²")),
            _row("Total Area", _fmt(vstab.get("vstab_area_total"), ".3f", suffix=" m²")),
            _row("Aspect Ratio", _fmt(vstab.get("vstab_AR"), ".2f")),
            _row("Volume Coefficient", _fmt(vstab.get("V_V_actual"), ".4f")),
            _row("Airfoil", _h(str(vstab.get("vstab_airfoil", "N/A")))),
        ]))

    if power:
        power_rows = [
            _row("Solar Panels", _fmt(power.get("solar_panels_n"), ".0f")),
            _row("Battery Capacity", _fmt(power.get("battery_capacity"), ".1f", suffix=" Wh")),
        ]
        if power.get("solar_panel_side_length") is not None:
            power_rows.append(_row("Panel Size", _fmt(_num(power.get("solar_panel_side_length")) * 1000, ".1f", suffix=" mm")))
        parts.append(_section("Power System", power_rows))

        # Chart container if battery_states exist
        battery_states = power.get("battery_states") or []
        if isinstance(battery_states, list) and len(battery_states) > 0:
            parts.append(
                "<div style='margin-top: 15px;'>"
                "<h4 style='margin-bottom: 10px;'>Power Generation & Battery State</h4>"
                "<div style='position: relative; height: 200px; width: 100%; max-height: 200px; overflow: hidden;'>"
                "<canvas id='powerChart' style='max-height: 200px;'></canvas>"
                "</div>"
                "</div>"
            )

    if prop:
        prop_rows: List[str] = []
        if prop.get("propeller_diameter") is not None:
            prop_rows.append(_row("Propeller Diameter", _fmt(prop.get("propeller_diameter"), ".3f", suffix=" m")))
        if prop.get("propeller_n") is not None:
            prop_rows.append(_row("Number of Motors", _fmt(prop.get("propeller_n"), ".0f")))
        if prop_rows:
            parts.append(_section("Propulsion", prop_rows))

    if main_wing.get("struct_defined_aoa") is not None:
        parts.append(_section("Wing Configuration", [
            _row("Structural AOA", _fmt(main_wing.get("struct_defined_aoa"), ".2f", suffix="°")),
            _row("Mean Aerodynamic Chord", _fmt(main_wing.get("MAC_w"), ".3f", suffix=" m")),
            _row("Airfoil", _h(str(main_wing.get("wing_airfoil", "N/A")))),
        ]))

    if geom.get("boom_radius") is not None:
        parts.append(_section("Structural Details", [
            _row("Boom Radius", _fmt(_num(geom.get("boom_radius")) * 1000, ".1f", suffix=" mm")),
            _row("Boom Spacing Fraction", _fmt(geom.get("boom_spacing_frac"), ".2f")),
        ]))

    parts.append("</div>")
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
    body {{ font-family: Arial, sans-serif; overflow: hidden; background: #1a1a1a; }}
    #canvas-container {{ width: 100vw; height: 100vh; position: relative; }}

    #sidebar {{
      position: fixed; top: 10px; right: 10px;
      width: 350px; max-height: calc(100vh - 20px);
      background: rgba(245,245,245,0.95);
      padding: 15px; overflow-y: auto;
      border: 1px solid #ddd; border-radius: 5px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
      z-index: 1000;
    }}
    #sidebar h2 {{ margin-top: 0; color: #333; font-size: 18px; }}
    #sidebar h3 {{ margin-top: 15px; margin-bottom: 10px; color: #555; font-size: 14px; }}
    #sidebar table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }}
    #sidebar table td {{ padding: 5px; border-bottom: 1px solid #eee; }}
    #sidebar table td:first-child {{ font-weight: bold; width: 60%; }}

    #legend {{
      position: fixed; top: 10px; left: 10px;
      background: rgba(255,255,255,0.95);
      padding: 15px; border-radius: 5px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
      z-index: 1000;
      max-height: calc(100vh - 20px);
      overflow-y: auto;
      min-width: 200px;
    }}
    #legend h3 {{ margin-top: 0; margin-bottom: 10px; color: #333; }}
    .legend-item {{ display: flex; align-items: center; padding: 5px 0; cursor: pointer; user-select: none; }}
    .legend-item:hover {{ background: rgba(0,0,0,0.05); }}
    .legend-color {{ width: 20px; height: 20px; border: 1px solid #333; margin-right: 8px; flex-shrink: 0; }}
    .legend-item.hidden {{ opacity: 0.3; }}

    #info {{
      position: fixed; bottom: 10px; left: 10px;
      background: rgba(0,0,0,0.7);
      color: white; padding: 10px; border-radius: 5px;
      font-size: 12px; z-index: 1000;
    }}

    #tooltip {{
      position: fixed;
      background: rgba(0,0,0,0.9);
      color: white;
      padding: 10px 15px;
      border-radius: 5px;
      font-size: 12px;
      pointer-events: none;
      z-index: 2000;
      display: none;
      max-width: 300px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}
    #tooltip.visible {{ display: block; }}
    #tooltip h4 {{ margin: 0 0 5px 0; color: #4CAF50; font-size: 14px; }}
    #tooltip p {{ margin: 3px 0; }}
  </style>
</head>
<body>
  <div id="canvas-container"></div>

  <div id="legend">
    <h3>Components</h3>
    <div id="legend-content"></div>
  </div>

  <div id="sidebar">
    {specs_html}
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

    const batteryStates = soln.Power?.battery_states || null;
    const batteryCapacity = soln.Power?.battery_capacity || 0;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Align to expected aircraft view: rotate scene instead of remapping every component
    scene.rotation.z = Math.PI;
    scene.rotation.x = -Math.PI / 2;

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    function getColorForCategory(name) {{
      const n = name.toLowerCase();
      if (n.includes('wing') || n.includes('stabilizer')) return 0x4A90E2;
      if (n.includes('boom') || n.includes('fuselage') || n.includes('superstructure') || n.includes('interface')) return 0x8B7355;
      if (n.includes('motor') || n.includes('esc') || n.includes('prop')) return 0xE67E22;
      if (n.includes('batter')) return 0xE74C3C;
      if (n.includes('solar')) return 0xF1C40F;
      if (n.includes('fc') || n.includes('gps') || n.includes('telemetry') || n.includes('receiver') || n.includes('navlight') || n.includes('power board')) return 0x27AE60;
      return 0x9B59B6;
    }}

    function createMeshPhong(color, opacity=0.7) {{
      return new THREE.MeshPhongMaterial({{
        color, opacity, transparent: opacity < 1.0, side: THREE.DoubleSide
      }});
    }}

    function createBox(center, Lx, Ly, Lz, color, userData) {{
      const geometry = new THREE.BoxGeometry(Lx, Ly, Lz, 8, 8, 8);
      const material = createMeshPhong(color, 0.7);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(center[0], center[1], center[2]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = userData;
      mesh.userData.originalMaterial = material.clone();
      return mesh;
    }}

    function createCylinder(center, radius, length, axis, color, userData) {{
      const geometry = new THREE.CylinderGeometry(radius, radius, length, 64);
      const material = createMeshPhong(color, 0.7);
      const mesh = new THREE.Mesh(geometry, material);

      if (axis === 'x') mesh.rotation.z = Math.PI / 2;
      else if (axis === 'z') mesh.rotation.x = Math.PI / 2;

      mesh.position.set(center[0], center[1], center[2]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = userData;
      mesh.userData.originalMaterial = material.clone();
      return mesh;
    }}

    function createSphere(center, radius, color, userData) {{
      const geometry = new THREE.SphereGeometry(radius, 32, 32);
      const material = createMeshPhong(color, 0.8);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(center[0], center[1], center[2]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = userData;
      mesh.userData.originalMaterial = material.clone();
      return mesh;
    }}

    function reconstructGeometry(soln, massProperties) {{
      const mainWing = soln['Main Wing'] || soln.Main_Wing || {{}};
      const geometry = soln.Geometry || {{}};
      const hstab = soln.HStab || {{}};
      const vstab = soln['V Stab'] || soln.V_Stab || {{}};
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

      // Precompute maxMass safely
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
    const cgGeometry = new THREE.SphereGeometry(0.05, 32, 32);
    const cgMaterial = new THREE.MeshPhongMaterial({{ color: 0xff0000 }});
    const cgMesh = new THREE.Mesh(cgGeometry, cgMaterial);
    cgMesh.position.set(total.x_cg || 0, total.y_cg || 0, total.z_cg || 0);
    cgMesh.userData = {{
      name: 'Total CG', type: 'cg', mass: total.mass || 0,
      center: [total.x_cg || 0, total.y_cg || 0, total.z_cg || 0],
      originalMaterial: cgMaterial.clone()
    }};
    scene.add(cgMesh);
    componentMeshes['Total CG'] = cgMesh;

    // Wireframe tube helper (visible lines)
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

      for (const wing of airplaneGeometry.wings) {{
        const xsecs = wing.xsecs || [];
        if (xsecs.length < 2) continue;

        function le(xsec) {{
          return new THREE.Vector3(xsec.xyz_le[0] + xyz_ref[0], xsec.xyz_le[1] + xyz_ref[1], xsec.xyz_le[2] + xyz_ref[2]);
        }}
        function te(xsec) {{
          return new THREE.Vector3(xsec.xyz_le[0] + xsec.chord + xyz_ref[0], xsec.xyz_le[1] + xyz_ref[1], xsec.xyz_le[2] + xyz_ref[2]);
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
          const c1 = new THREE.Vector3(a.xyz_c[0] + xyz_ref[0], a.xyz_c[1] + xyz_ref[1], a.xyz_c[2] + xyz_ref[2]);
          const c2 = new THREE.Vector3(b.xyz_c[0] + xyz_ref[0], b.xyz_c[1] + xyz_ref[1], b.xyz_c[2] + xyz_ref[2]);
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

    // Hover tooltip + highlight
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const tooltip = document.getElementById('tooltip');
    const allMeshes = Object.values(componentMeshes);

    function restoreMaterial(mesh) {{
      if (mesh?.userData?.originalMaterial) {{
        mesh.material.dispose();
        mesh.material = mesh.userData.originalMaterial.clone();
      }}
    }}

    function highlightMesh(mesh) {{
      if (!mesh?.material) return;
      const m = mesh.material.clone();
      m.emissive = new THREE.Color(0xffffff);
      m.emissiveIntensity = 0.5;
      mesh.material = m;
    }}

    let hovered = null;

    function onMouseMove(event) {{
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(allMeshes, true);

      if (hits.length) {{
        const obj = hits[0].object;
        if (hovered !== obj) {{
          if (hovered) restoreMaterial(hovered);
          hovered = obj;
          highlightMesh(obj);

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
        }}
      }} else {{
        if (hovered) {{
          restoreMaterial(hovered);
          hovered = null;
        }}
        tooltip.classList.remove('visible');
      }}
    }}

    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('mouseleave', () => {{
      if (hovered) restoreMaterial(hovered);
      hovered = null;
      tooltip.classList.remove('visible');
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
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
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                yAxisID: 'y',
                tension: 0.1,
                pointRadius: 0
              }},
              {{
                label: 'Power Generation (W)',
                data: powerGeneration,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
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
              title: {{ display: true, text: '24-Hour Power Generation & Battery State' }},
              legend: {{ display: true, position: 'top' }}
            }},
            scales: {{
              x: {{ display: true, title: {{ display: true, text: 'Time (hours)' }} }},
              y: {{
                type: 'linear', display: true, position: 'left',
                title: {{ display: true, text: 'Battery State (Wh)' }},
                min: 0,
                max: batteryCapacity ? batteryCapacity * 1.1 : undefined
              }},
              y1: {{
                type: 'linear', display: true, position: 'right',
                title: {{ display: true, text: 'Power Generation (W)' }},
                grid: {{ drawOnChartArea: false }}
              }}
            }}
          }}
        }});
      }}
    }}
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
        output_path = soln_path.parent / "ghost.html"

    soln = json.loads(soln_path.read_text(encoding="utf-8"))
    mass_properties = json.loads(mass_properties_path.read_text(encoding="utf-8"))

    specs_html = format_specs_html(soln, mass_properties)

    airplane_geometry: JSONDict = {}
    if airplane_path and airplane_path.suffix.lower() == ".aero" and airplane_path.exists():
        airplane_geometry = extract_airplane_geometry(airplane_path)

    # Embed compact JSON to keep HTML size reasonable.
    soln_json = json.dumps(soln, separators=(",", ":"))
    mass_properties_json = json.dumps(mass_properties, separators=(",", ":"))
    airplane_geometry_json = json.dumps(airplane_geometry, separators=(",", ":"))

    html_content = HTML_TEMPLATE.format(
        specs_html=specs_html,
        soln_json=soln_json,
        mass_properties_json=mass_properties_json,
        airplane_geometry_json=airplane_geometry_json,
    )

    output_path.write_text(html_content, encoding="utf-8")
    print(f"[viewer] Saved Three.js visualization: {output_path}")
    return output_path


# ----------------------------
# CLI
# ----------------------------

def resolve_inputs(
    arg: Optional[str],
) -> Tuple[Path, Path, Optional[Path], Path]:
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

    # Prefer .aero if available (wireframe), but do not assume .step for this pipeline.
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

    out_html = create_threejs_viewer(soln_path, mass_properties_path, airplane_path, run_dir / "aircraft_mass_viewer.html")
    print(f"[viewer] Complete! Open {out_html} in a browser.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import aerosandbox as asb

PathLike = Union[str, Path]


def export_xflr5_xml_from_soln(
    *,
    airplane_sol: asb.Airplane,
    soln: Dict[str, Any],
    out_path: PathLike,
    include_fuselages: bool = False,
) -> None:
    """Exports an XFLR5 XML from an already-solved airplane, using parameters stored in `soln`.

    AeroSandbox's XFLR5 exporter supports exactly 3 lifting surfaces: main wing, elevator, and fin.
    This project has a twin-fin configuration, so we synthesize a single centerline fin using
    values from the grouped output sections (e.g. `soln['Geometry']`, `soln['V Stab']`, `soln['HStab']`).
    """
    out_path = Path(out_path)

    geom = soln["Geometry"]
    vstab = soln["V Stab"]
    hstab = soln["HStab"]

    boom_length = float(geom["boom_length"])
    vstab_root_chord = float(vstab["vstab_root_chord"])
    vstab_span = float(vstab["vstab_span"])
    vstab_tip_chord = float(hstab["hstab_chordlen"])

    tail_airfoil_name = soln.get("Airfoils", {}).get("tail_airfoil", "naca0010")
    tail_airfoil = asb.Airfoil(tail_airfoil_name)

    if len(getattr(airplane_sol, "wings", [])) < 2:
        raise ValueError("airplane_sol must have at least 2 wings (main wing, elevator).")

    mainwing_xflr = airplane_sol.wings[0]
    elevator_xflr = airplane_sol.wings[1]

    fin_xflr = asb.Wing(
        name="Vertical Stabilizer (XFLR5)",
        symmetric=False,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0.0, 0.0, 0.0],
                chord=vstab_root_chord,
                twist=0.0,
                airfoil=tail_airfoil,
            ),
            asb.WingXSec(
                xyz_le=[vstab_root_chord / 4, 0.0, vstab_span],
                chord=vstab_tip_chord,
                twist=0.0,
                airfoil=tail_airfoil,
            ),
        ],
    ).translate([boom_length, 0.0, 0.0])

    airplane_sol.export_XFLR5_xml(
        str(out_path),
        include_fuselages=include_fuselages,
        mainwing=mainwing_xflr,
        elevator=elevator_xflr,
        fin=fin_xflr,
    )


def export_cadquery_step(
    *,
    airplane_sol: asb.Airplane,
    out_path: PathLike,
    minimum_airfoil_TE_thickness: float = 0.001,
) -> None:
    """Exports a CadQuery STEP file from an already-solved airplane."""
    out_path = Path(out_path)
    airplane_sol.export_cadquery_geometry(
        str(out_path),
        minimum_airfoil_TE_thickness=float(minimum_airfoil_TE_thickness),
    )


#!/usr/bin/env python3
"""
48-hour energy balance analysis for a solar aircraft in circular flight.

Reads aircraft/mission parameters from an artifact folder containing soln.json,
computes solar generation vs. power consumption (including turn load factor),
integrates battery state, classifies day/dawn-dusk/night, and exports plots + data.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from aerosandbox.library import power_solar

from lib.artifacts import latest_run_dir


# Workaround for some AeroSandbox installs where `aerosandbox.library.power_solar` forgets to import Atmosphere.
# (Seen as: NameError: name 'Atmosphere' is not defined inside power_solar.airmass()).
try:
    from aerosandbox.atmosphere import Atmosphere as _Atmosphere
    power_solar.Atmosphere = _Atmosphere
except Exception:
    # If AeroSandbox internals change, fail gracefully; downstream call will surface a clearer error.
    pass


# ----------------------------
# Data model / configuration
# ----------------------------

@dataclass(frozen=True)
class ArtifactData:
    # Mission
    mission_date: int = 100
    operating_lat: float = 37.398928
    operating_altitude: float = 1200.0

    # Performance
    airspeed: float = 10.0  # m/s
    power_cruise_all_motors: float = 50.0  # W

    # Power system
    solar_panels_n: int = 50
    battery_capacity_Wh: float = 400.0
    solar_panel_side_length_m: float = 0.125
    solar_cell_efficiency: float = 0.2187
    energy_generation_margin: float = 1.05
    allowable_battery_depth_of_discharge: float = 0.85

    # Avionics
    avionics_power_W: float = 9.0  # 8W avionics + 1W navlights


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    """Helper: get key from dict with fallback."""
    v = d.get(key, default)
    return default if v is None else v


def load_artifact_data(artifact_path: Path) -> ArtifactData:
    soln_path = artifact_path / "soln.json"
    if not soln_path.exists():
        raise FileNotFoundError(f"Solution file not found: {soln_path}")

    with soln_path.open("r") as f:
        soln = json.load(f)

    mission = soln.get("Mission", {})
    performance = soln.get("Performance", {})
    power = soln.get("Power", {})

    return ArtifactData(
        mission_date=int(_get(mission, "mission_date", 100)),
        operating_lat=float(_get(mission, "operating_lat", 37.398928)),
        operating_altitude=float(_get(mission, "operating_altitude", 1200.0)),
        airspeed=float(_get(performance, "airspeed", 10.0)),
        power_cruise_all_motors=float(_get(performance, "power_cruise (all motors)", 50.0)),
        solar_panels_n=int(_get(power, "solar_panels_n", 50)),
        battery_capacity_Wh=float(_get(power, "battery_capacity", 400.0)),
        solar_panel_side_length_m=float(_get(power, "solar_panel_side_length", 0.125)),
        solar_cell_efficiency=float(_get(power, "solar_cell_efficiency", 0.2187)),
        energy_generation_margin=float(_get(power, "energy_generation_margin", 1.05)),
        allowable_battery_depth_of_discharge=float(_get(power, "allowable_battery_depth_of_discharge", 0.85)),
        avionics_power_W=9.0,
    )


# ----------------------------
# Physics / computation
# ----------------------------

def calculate_turn_power_W(power_straight_W: float, airspeed_mps: float, turn_radius_m: float) -> float:
    """
    Coordinated turn load factor n = sqrt(1 + (V^2/(gR))^2)
    Power ~ n^1.5 (approx drag scaling).
    """
    g = 9.81
    if turn_radius_m <= 0:
        raise ValueError("turn_radius_m must be > 0")

    n = np.sqrt(1.0 + (airspeed_mps**2 / (g * turn_radius_m)) ** 2)
    return power_straight_W * (n ** 1.5)


def classify_periods(solar_flux_Wm2: np.ndarray) -> np.ndarray:
    """
    Classify time points into 'night', 'dawn_dusk', 'day' based on solar flux.
    """
    max_flux = float(np.max(solar_flux_Wm2)) if solar_flux_Wm2.size else 0.0
    night_threshold = max(1.0, 0.01 * max_flux)
    dawn_dusk_threshold = 0.20 * max_flux

    # np.select is clearer than nested np.where for 3+ cases
    return np.select(
        [
            solar_flux_Wm2 < night_threshold,
            solar_flux_Wm2 < dawn_dusk_threshold,
        ],
        [
            "night",
            "dawn_dusk",
        ],
        default="day",
    )


def compute_energy_balance(
    a: ArtifactData,
    circular_diameter_m: float,
    n_points: int = 360,
    duration_hours: float = 48.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Returns:
        time_hours, solar_flux, power_generated, power_used, net_energy_Wh,
        battery_state_Wh, period_type, battery_min_threshold_Wh, battery_capacity_Wh
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if circular_diameter_m <= 0:
        raise ValueError("circular_diameter_m must be > 0")

    # Time grid
    time_s = np.linspace(0.0, duration_hours * 3600.0, n_points)
    time_hours = time_s / 3600.0
    dt_hours = (time_s[1] - time_s[0]) / 3600.0

    # Turn power
    turn_radius_m = 0.5 * circular_diameter_m
    turn_power_W = calculate_turn_power_W(a.power_cruise_all_motors, a.airspeed, turn_radius_m)
    power_used_W = float(turn_power_W + a.avionics_power_W)  # constant in this scenario

    # Solar area
    solar_area_m2 = a.solar_panels_n * (a.solar_panel_side_length_m ** 2)

    # Solar flux vectorization:
    # Compute time-of-day in seconds for each point (repeat each 24h)
    time_of_day_s = np.mod(time_s, 24.0 * 3600.0).astype(float)

    # Call Aerosandbox solar_flux for each time point.
    # If solar_flux supports numpy arrays for time, this will be fast;
    # if it only supports scalars, np.vectorize will still simplify code.
    try:
        flux = power_solar.solar_flux(
            latitude=a.operating_lat,
            day_of_year=a.mission_date,
            time=time_of_day_s,
            altitude=a.operating_altitude,
            panel_azimuth_angle=0,
            panel_tilt_angle=0,
        )
        solar_flux_Wm2 = np.asarray(flux, dtype=float)
    except Exception:
        # Fallback: safe vectorization for scalar-only APIs
        solar_flux_Wm2 = np.vectorize(
            lambda t: power_solar.solar_flux(
                latitude=a.operating_lat,
                day_of_year=a.mission_date,
                time=float(t),
                altitude=a.operating_altitude,
                panel_azimuth_angle=0,
                panel_tilt_angle=0,
            ),
            otypes=[float],
        )(time_of_day_s)

    solar_flux_Wm2 = np.nan_to_num(solar_flux_Wm2, nan=0.0, posinf=0.0, neginf=0.0)
    solar_flux_Wm2 = np.maximum(solar_flux_Wm2, 0.0)

    # Power generated
    power_generated_W = solar_flux_Wm2 * solar_area_m2 * a.solar_cell_efficiency / a.energy_generation_margin

    # Net energy per step (Wh)
    power_used_series_W = np.full_like(power_generated_W, power_used_W)
    net_energy_Wh = (power_generated_W - power_used_series_W) * dt_hours

    # Battery: +7% oversizing, clamp at max only
    battery_capacity_Wh = a.battery_capacity_Wh * 1.07
    # Start at full capacity, then accumulate net energy changes
    battery_state_Wh = np.zeros_like(net_energy_Wh)
    battery_state_Wh[0] = battery_capacity_Wh
    for i in range(1, len(battery_state_Wh)):
        battery_state_Wh[i] = battery_state_Wh[i-1] + net_energy_Wh[i-1]
        # Only clamp to maximum, allow going below minimum
        battery_state_Wh[i] = min(battery_state_Wh[i], battery_capacity_Wh)

    battery_min_threshold_Wh = battery_capacity_Wh * (1.0 - a.allowable_battery_depth_of_discharge)

    period_type = classify_periods(solar_flux_Wm2)

    return (
        time_hours,
        solar_flux_Wm2,
        power_generated_W,
        power_used_series_W,
        net_energy_Wh,
        battery_state_Wh,
        period_type,
        float(battery_min_threshold_Wh),
        float(battery_capacity_Wh),
    )


# ----------------------------
# Plotting / export
# ----------------------------

def _add_period_shading(ax, time_hours, period_type, y_min, y_max):
    # Masks
    night = period_type == "night"
    dawn = period_type == "dawn_dusk"
    day = period_type == "day"

    # Night
    if np.any(night):
        ax.fill_between(time_hours, y_min, y_max, where=night, alpha=0.20, color="#000080",
                        label="Night", interpolate=True, zorder=0)

    # Dawn/Dusk
    if np.any(dawn):
        ax.fill_between(time_hours, y_min, y_max, where=dawn, alpha=0.18, color="#FFA500",
                        label="Dawn/Dusk", interpolate=True, zorder=0)

    # Day (no label to avoid legend clutter)
    if np.any(day):
        ax.fill_between(time_hours, y_min, y_max, where=day, alpha=0.15, color="#FFFACD",
                        label="", interpolate=True, zorder=0)

    # 24h marker
    ax.axvline(x=24, color="gray", linestyle="--", alpha=0.5, linewidth=1, zorder=10)
    ax.text(
        24, y_max * 0.98, "Day 2", rotation=90, va="top", ha="right", fontsize=9, alpha=0.7, zorder=20,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )


def plot_results(
    time_hours: np.ndarray,
    solar_flux_Wm2: np.ndarray,
    power_generated_W: np.ndarray,
    power_used_W: np.ndarray,
    battery_state_Wh: np.ndarray,
    period_type: np.ndarray,
    battery_min_threshold_Wh: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle("48-Hour (2-Day) Energy Balance Analysis", fontsize=16, fontweight="bold")

    # y-ranges (for shading)
    battery_y_min, battery_y_max = 0.0, float(np.max(battery_state_Wh)) * 1.05
    power_y_min = float(min(np.min(power_generated_W), np.min(power_used_W))) * 0.95
    power_y_max = float(max(np.max(power_generated_W), np.max(power_used_W))) * 1.05
    flux_y_min, flux_y_max = 0.0, float(np.max(solar_flux_Wm2)) * 1.05

    overall_y_min = min(battery_y_min, power_y_min, flux_y_min)
    overall_y_max = max(battery_y_max, power_y_max, flux_y_max)

    _add_period_shading(ax, time_hours, period_type, overall_y_min, overall_y_max)

    # Battery-bad region
    below_min = battery_state_Wh < battery_min_threshold_Wh
    if np.any(below_min):
        ax.fill_between(
            time_hours, overall_y_min, overall_y_max,
            where=below_min, alpha=0.25, color="red",
            label="BELOW MINIMUM (unsafe)", interpolate=True, zorder=1
        )

    # Battery on left axis
    ax.plot(time_hours, battery_state_Wh, "b-", linewidth=2.5, label="Battery Energy (Wh)", zorder=6)
    ax.axhline(battery_min_threshold_Wh, color="r", linestyle="--", alpha=0.5, linewidth=1.5,
               label="Minimum Safe Level", zorder=6)
    ax.set_ylabel("Battery Energy (Wh)", color="blue", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.set_ylim(battery_y_min, battery_y_max)

    # Power + flux on right axis
    ax2 = ax.twinx()
    ax2.plot(time_hours, power_generated_W, "g-", linewidth=2, label="Power Generated (W)", zorder=5)
    ax2.plot(time_hours, power_used_W, "r-", linewidth=2, label="Power Used (W)", zorder=5)

    surplus = power_generated_W >= power_used_W
    deficit = ~surplus
    if np.any(surplus):
        ax2.fill_between(time_hours, power_generated_W, power_used_W, where=surplus, alpha=0.2, color="green",
                         label="Surplus", zorder=3)
    if np.any(deficit):
        ax2.fill_between(time_hours, power_generated_W, power_used_W, where=deficit, alpha=0.2, color="red",
                         label="Deficit", zorder=3)

    ax2.plot(time_hours, solar_flux_Wm2, "orange", linewidth=2, linestyle="--",
             label="Solar Flux (W/m²)", zorder=5)

    ax2.set_ylabel("Power (W) / Solar Flux (W/m²)", color="green", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(0.0, max(power_y_max, flux_y_max))

    ax.set_xlabel("Time (hours)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=1)

    # Legend: merge handles and drop empty labels
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles, labels = [], []
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)):
        if l and l not in labels:
            handles.append(h)
            labels.append(l)
    ax.legend(handles, labels, loc="upper left", fontsize=9)

    plt.tight_layout()
    plot_path = output_dir / "energy_balance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved plot to: {plot_path}")
    plt.show()


def export_data(
    time_hours: np.ndarray,
    period_type: np.ndarray,
    solar_flux_Wm2: np.ndarray,
    power_generated_W: np.ndarray,
    power_used_W: np.ndarray,
    net_energy_Wh: np.ndarray,
    battery_state_Wh: np.ndarray,
    artifact_data: ArtifactData,
    circular_diameter_m: float,
    battery_capacity_Wh: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "energy_balance.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_hours", "period_type", "solar_flux_W_per_m2", "power_generated_W",
            "power_used_W", "net_energy_Wh", "battery_state_Wh"
        ])
        for i in range(time_hours.size):
            w.writerow([
                float(time_hours[i]),
                str(period_type[i]),
                float(solar_flux_Wm2[i]),
                float(power_generated_W[i]),
                float(power_used_W[i]),
                float(net_energy_Wh[i]),
                float(battery_state_Wh[i]),
            ])
    print(f"[export] Saved CSV to: {csv_path}")

    # Summary metrics (trapz over hours gives Wh)
    total_gen_Wh = float(np.trapz(power_generated_W, time_hours))
    total_use_Wh = float(np.trapz(power_used_W, time_hours))
    surplus_Wh = total_gen_Wh - total_use_Wh

    day1 = time_hours <= 24.0
    day2 = time_hours > 24.0

    day1_gen = float(np.trapz(power_generated_W[day1], time_hours[day1]))
    day1_use = float(np.trapz(power_used_W[day1], time_hours[day1]))
    day2_gen = float(np.trapz(power_generated_W[day2], time_hours[day2]))
    day2_use = float(np.trapz(power_used_W[day2], time_hours[day2]))

    # Period stats
    dt_hours = float(time_hours[1] - time_hours[0])
    is_day = period_type == "day"
    is_dawn = period_type == "dawn_dusk"
    is_night = period_type == "night"

    def hours(mask: np.ndarray) -> float:
        return float(np.sum(mask) * dt_hours)

    battery_at_24h = float(battery_state_Wh[np.argmin(np.abs(time_hours - 24.0))])

    summary = {
        "analysis_parameters": {
            "circular_diameter_m": float(circular_diameter_m),
            "turn_radius_m": float(circular_diameter_m / 2.0),
            "n_points": int(time_hours.size),
            "simulation_duration_hours": float(time_hours[-1]),
        },
        "aircraft_parameters": artifact_data.__dict__,
        "period_statistics": {
            "total_daytime_hours": hours(is_day),
            "total_dawn_dusk_hours": hours(is_dawn),
            "total_nighttime_hours": hours(is_night),
            "day1_daytime_hours": hours(is_day & day1),
            "day1_dawn_dusk_hours": hours(is_dawn & day1),
            "day1_nighttime_hours": hours(is_night & day1),
            "day2_daytime_hours": hours(is_day & day2),
            "day2_dawn_dusk_hours": hours(is_dawn & day2),
            "day2_nighttime_hours": hours(is_night & day2),
        },
        "energy_metrics": {
            "total_energy_generated_Wh": total_gen_Wh,
            "total_energy_used_Wh": total_use_Wh,
            "energy_surplus_Wh": float(surplus_Wh),
            "energy_surplus_percent": float(100.0 * surplus_Wh / total_use_Wh) if total_use_Wh > 0 else 0.0,
        },
        "day1_metrics": {
            "energy_generated_Wh": day1_gen,
            "energy_used_Wh": day1_use,
            "energy_surplus_Wh": float(day1_gen - day1_use),
            "energy_surplus_percent": float(100.0 * (day1_gen - day1_use) / day1_use) if day1_use > 0 else 0.0,
        },
        "day2_metrics": {
            "energy_generated_Wh": day2_gen,
            "energy_used_Wh": day2_use,
            "energy_surplus_Wh": float(day2_gen - day2_use),
            "energy_surplus_percent": float(100.0 * (day2_gen - day2_use) / day2_use) if day2_use > 0 else 0.0,
        },
        "battery_metrics": {
            "battery_capacity_Wh": float(battery_capacity_Wh),
            "initial_state_Wh": float(battery_state_Wh[0]),
            "final_state_Wh": float(battery_state_Wh[-1]),
            "min_state_Wh": float(np.min(battery_state_Wh)),
            "max_state_Wh": float(np.max(battery_state_Wh)),
            "state_change_Wh": float(battery_state_Wh[-1] - battery_state_Wh[0]),
            "battery_at_24h_Wh": battery_at_24h,
        },
        "power_metrics": {
            "avg_power_generated_W": float(np.mean(power_generated_W)),
            "max_power_generated_W": float(np.max(power_generated_W)),
            "avg_power_used_W": float(np.mean(power_used_W)),
            "max_power_used_W": float(np.max(power_used_W)),
        },
    }

    json_path = output_dir / "energy_balance_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[export] Saved JSON summary to: {json_path}")

    # Console summary
    print("\n" + "=" * 70)
    print("48-HOUR (2-DAY) ENERGY BALANCE SUMMARY")
    print("=" * 70)
    print(f"Total Energy Generated: {total_gen_Wh:.2f} Wh")
    print(f"Total Energy Used:      {total_use_Wh:.2f} Wh")
    print(f"Energy Surplus:         {surplus_Wh:.2f} Wh ({summary['energy_metrics']['energy_surplus_percent']:.1f}%)")
    print(f"Battery: initial={battery_state_Wh[0]:.2f} Wh, at24h={battery_at_24h:.2f} Wh, "
          f"final={battery_state_Wh[-1]:.2f} Wh, min={np.min(battery_state_Wh):.2f} Wh")
    print("=" * 70)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="48-hour energy balance for solar aircraft in circular flight")
    p.add_argument("--artifact-path", type=str, default=None,
                   help="Path to artifact folder containing soln.json (default: latest run in output/)")
    p.add_argument("--circular-diameter", type=float, default=1000.0,
                   help="Diameter of circular flight pattern in meters (default: 1000)")
    p.add_argument("--n-points", type=int, default=3000,
                   help="Number of time discretization points (default: 3000 for 2 days)")
    args = p.parse_args()

    if args.artifact_path:
        artifact_path = Path(args.artifact_path)
    else:
        artifact_root = Path(__file__).parent / "output"
        artifact_path = latest_run_dir(artifact_root)
        print(f"[config] Using latest run: {artifact_path}")

    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {artifact_path}")

    # Output directory is the same as artifact directory
    output_dir = artifact_path

    print(f"[load] Loading artifact data from: {artifact_path}")
    a = load_artifact_data(artifact_path)

    print(f"[compute] Computing 48h energy balance (diameter={args.circular_diameter} m, n_points={args.n_points})")
    (
        time_hours,
        solar_flux,
        power_generated,
        power_used,
        net_energy,
        battery_state,
        period_type,
        battery_min_threshold,
        battery_capacity,
    ) = compute_energy_balance(a, args.circular_diameter, n_points=args.n_points)

    print("[plot] Generating plot...")
    plot_results(
        time_hours=time_hours,
        solar_flux_Wm2=solar_flux,
        power_generated_W=power_generated,
        power_used_W=power_used,
        battery_state_Wh=battery_state,
        period_type=period_type,
        battery_min_threshold_Wh=battery_min_threshold,
        output_dir=output_dir,
    )

    print("[export] Exporting CSV and JSON...")
    export_data(
        time_hours=time_hours,
        period_type=period_type,
        solar_flux_Wm2=solar_flux,
        power_generated_W=power_generated,
        power_used_W=power_used,
        net_energy_Wh=net_energy,
        battery_state_Wh=battery_state,
        artifact_data=a,
        circular_diameter_m=args.circular_diameter,
        battery_capacity_Wh=battery_capacity,
        output_dir=output_dir,
    )

    print(f"\n[complete] Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

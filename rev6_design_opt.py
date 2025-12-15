import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.library import power_solar, propulsion_electric, propulsion_propeller
from aerosandbox.atmosphere import Atmosphere
import numpy as onp


opti=asb.Opti(cache_filename="output/soln1.json")


### CONSTANTS
## Physics
g = 9.81

## Mission
mission_date = 100
operating_lat = 37.398928
togw_max = 12 # kg
temperature_high = 35 # 35°K hotter than ISA --> leads to ~115 °F operating temperature
operating_altitude = 1200 # in meters
operating_atm = Atmosphere(operating_altitude, temperature_deviation=temperature_high)

## Performance
min_climb_angle = 30 # degrees

## Aerodynamics
# Airfoils
wing_airfoil = asb.Airfoil("s4110")
tail_airfoil = asb.Airfoil("naca0010")

# Main wing
polyhedral_angle = 5

# Tail sizing (via Tail Volume Coefficients)
V_H = 0.50  # Horizontal tail volume coefficient
V_V = 0.02  # Vertical tail volume coefficient

# Static margin constraints (dimensionless, SM = (x_np - x_cg) / MAC)
SM_min = 0.05
SM_max = 0.5

# Stall / low-Re realism (AeroBuildup does not model stall)
CLmax_cruise = 1.3          # conservative-ish small UAV CLmax (update if you have airfoil data)
stall_speed_margin = 1.2    # V_cruise >= margin * V_stall

## Structural
structural_mass_markup = 1.2
boom_radius = 0.01
boom_spacing_frac = 0.25

## Propulsion
eta_prop = 0.90

## Power
battery_voltage = 22.2
N = 180  # Number of discretization points
time = onp.linspace(0, 24 * 60 * 60, N)  # s (numeric; does not need to be symbolic)
dt = onp.diff(time)[0]  # s
solar_panel_side_length = 0.125 # m
solar_panels_n_rows = 2
solar_encapsulation_eff_hit = 0.1 # Estimated 10% efficieincy loss from encapsulation.
solar_cell_efficiency = 0.243 * (1 - solar_encapsulation_eff_hit)
energy_generation_margin = 1.05
allowable_battery_depth_of_discharge = 0.85  # How much of the battery can you actually use?
pack_energy_Wh = 5 * 6 * 3.7  # 5 Ah, 6 cells, 3.7 V/cell

# Precompute solar flux profile numerically (robust vs NaNs from symbolic trig/acos edge cases)
solar_flux_profile = onp.array(
    [
        power_solar.solar_flux(
            latitude=operating_lat,
            day_of_year=mission_date,
            time=float(t),
            altitude=operating_altitude,
            panel_azimuth_angle=0,
            panel_tilt_angle=0,
        )
        for t in time
    ],
    dtype=float,
)
solar_flux_profile = onp.nan_to_num(solar_flux_profile, nan=0.0, posinf=0.0, neginf=0.0)
solar_flux_profile = onp.maximum(solar_flux_profile, 0.0)
print(
    f"[solar_flux_profile] min={solar_flux_profile.min():.3f} W/m^2, "
    f"max={solar_flux_profile.max():.3f} W/m^2, "
    f"nan_count={onp.isnan(solar_flux_profile).sum()}"
)



### VARIABLES
## Performance
airspeed = opti.variable(init_guess=15, lower_bound=5, upper_bound=30, scale=5, category="airspeed")
togw_design = opti.variable(init_guess=4, lower_bound=1e-3, upper_bound=togw_max, category="togw_max")
power_out_max = opti.variable(init_guess=500, lower_bound=25*16, scale=100, category="power_out_max")

## Propulsion
thrust_cruise = opti.variable(init_guess=4, lower_bound=0, scale=2, category="thrust_cruise")
propeller_n = opti.parameter(2)
propeller_diameter = opti.variable(init_guess=0.5, lower_bound=0.1, upper_bound=2, scale=1, category="propeller_diameter")

## Avionics
solar_panels_n = opti.variable(init_guess=40, lower_bound=10, category="solar_panels_n", scale=40)
battery_capacity = opti.variable(init_guess=450, lower_bound=100, category="battery_capacity", scale=150)  # initial battery energy in Wh
battery_states = opti.variable(n_vars=N, init_guess=500, category="battery_states", scale=100)

## Aerodynamics
# Main wing
wingspan = opti.variable(init_guess=6, lower_bound=2, upper_bound=7, scale=2, category="wingspan")
chordlen = opti.variable(init_guess=0.3, lower_bound=0.05, scale=1, category="chordlen")
struct_defined_aoa = opti.variable(init_guess=2, lower_bound=0, upper_bound=7, scale=1, category="struct_aoa")
cg_le_dist = 0.25 * chordlen  # CG assumed at quarter-chord of main wing
# alpha_cruise = opti.variable(init_guess=3, lower_bound=-2, upper_bound=10, scale=5, category="alpha_cruise")

# Empennage (sized by tail volume coeffs)
hstab_AR = opti.variable(init_guess=6.0, lower_bound=2.0, upper_bound=20.0, scale=5, category="hstab_AR")
vstab_AR = 2.0
hstab_aoa = opti.variable(init_guess=-5, lower_bound=-10, upper_bound=5, scale=5, category="hstab_aoa")

# Structural
boom_length = opti.variable(init_guess=0.5, lower_bound=0.1, upper_bound=2, scale=0.5, category="boom_length")
boom_y = 0.5 * boom_spacing_frac * wingspan



### GEOMETRIES
main_wing = asb.Wing(
    name="Main Wing",
    symmetric=True,  # Should this wing be mirrored across the XZ plane?
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=[
                0,
                0,
                0,
            ], # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=chordlen,
            twist=struct_defined_aoa,
            airfoil=wing_airfoil,
        ),
        asb.WingXSec(  # Mid
            xyz_le=[0.00, boom_y, 0],
            chord=chordlen,
            twist=struct_defined_aoa,
            airfoil=wing_airfoil,
        ),
        asb.WingXSec(  # Tip
            xyz_le=[0.00, wingspan / 2, np.sin(10 * np.pi / 180) * 0.5 * wingspan / 2],
            chord=chordlen / 2,
            twist=struct_defined_aoa,
            airfoil=wing_airfoil,
        ),
    ],
)

# Tail sizing from volume coefficients + aspect ratio (rectangular planforms)
S_w = main_wing.area()
MAC_w = main_wing.mean_aerodynamic_chord()
b_w = main_wing.span()

# Horizontal tail
L_H = boom_length - cg_le_dist
hstab_area = V_H * S_w * MAC_w / L_H
hstab_span = boom_y * 2
hstab_chordlen = hstab_area / hstab_span

# Vertical tail
L_V = boom_length - cg_le_dist
vstab_area_total = V_V * S_w * b_w / L_V
vstab_area = 0.5 * vstab_area_total
vstab_span = np.sqrt(vstab_AR * vstab_area)  # "span" here is height
vstab_root_chord = (2 * vstab_area / vstab_span) - hstab_chordlen  # trapezoid area: S = b*(cr+ct)/2

# Model horizontal and vertical stabilizers.
hor_stabilizer = asb.Wing(
    name="Horizontal Stabilizer",
    symmetric=True,
    xsecs=[
        asb.WingXSec(  # root
            xyz_le=[0, 0, 0],
            chord=hstab_chordlen,
            twist=hstab_aoa,
            airfoil=tail_airfoil,
        ),
        asb.WingXSec(  # tip
            xyz_le=[0.0, hstab_span / 2, 0],
            chord=hstab_chordlen,
            twist=hstab_aoa,
            airfoil=tail_airfoil,
        ),
    ],
).translate([boom_length + vstab_root_chord/4, 0, vstab_span])

vert_stabilizer_L = asb.Wing(
    name="Vertical Stabilizer L",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=vstab_root_chord,
            twist=0,
            airfoil=tail_airfoil,
        ),
        asb.WingXSec(
            xyz_le=[vstab_root_chord/4, 0, vstab_span],
            chord=hstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
    ],
).translate([boom_length, boom_y, 0])

vert_stabilizer_R = asb.Wing(
    name="Vertical Stabilizer R",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=vstab_root_chord,
            twist=0,
            airfoil=tail_airfoil,
        ),
        asb.WingXSec(
            xyz_le=[vstab_root_chord/4, 0, vstab_span],
            chord=hstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
    ],
).translate([boom_length, -boom_y, 0])

# Booms
boom_L = asb.Fuselage(
    name="Boom L",
    xsecs=[
        asb.FuselageXSec(xyz_c=[boom_length * xi, boom_y, -0.02], radius=boom_radius)
        for xi in np.cosspace(0, 1, 20)
    ],
)
boom_R = asb.Fuselage(
    name="Boom R",
    xsecs=[
        asb.FuselageXSec(xyz_c=[boom_length * xi, -boom_y, -0.02], radius=boom_radius)
        for xi in np.cosspace(0, 1, 20)
    ],
)


left_pod = asb.Fuselage(  # left pod fuselage
    name="Left Fuse",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0.5 * xi - 0.5, boom_y, -0.02],
            radius=0.4
            * asb.Airfoil("dae51").local_thickness(
                x_over_c=xi
            ),  # half a meter fuselage. Starting at LE and 0.5m forward
        )
        for xi in np.cosspace(0, 1, 30)
    ],
)

right_pod = asb.Fuselage(  # right pod fuselage
    name="Right Fuselage",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0.5 * xi - 0.5, -boom_y, -0.02],
            radius=0.4
            * asb.Airfoil("dae51").local_thickness(
                x_over_c=xi
            ),  # half a meter fuselage. Starting at LE and 0.5m forward
        )
        for xi in np.cosspace(0, 1, 30)
    ],
)

# Model the airplane.
airplane = asb.Airplane(
    name="Rev 6",
    xyz_ref=[cg_le_dist, 0, 0],  # CG location (measured from main wing LE)
    wings=[main_wing, hor_stabilizer, vert_stabilizer_L, vert_stabilizer_R],
    fuselages=[left_pod, right_pod, boom_L, boom_R],
)


### AERODYNAMICS
vlm = asb.AeroBuildup(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        atmosphere=operating_atm,
        velocity=airspeed,  # m/s
        # alpha=alpha_cruise,  # deg (trim variable)
    ),
)
aero = vlm.run_with_stability_derivatives(
    alpha=True,
    beta=True,
    p=False,
    q=False,
    r=False
)
aero["power"] = aero["D"] * airspeed



### STABILITY
static_margin = (aero["x_np"] - cg_le_dist) / main_wing.mean_aerodynamic_chord()


### PROPULSION
power_shaft_cruise = propulsion_propeller.propeller_shaft_power_from_thrust(
    thrust_force=thrust_cruise,
    area_propulsive=np.pi / 4 * propeller_diameter ** 2,
    airspeed=airspeed,
    rho=operating_atm.density(),
    propeller_coefficient_of_performance=0.90  # calibrated to QProp output with Dongjoon
)

propeller_tip_mach = 0.36  # From Dongjoon, 4/30/20
propeller_rads_per_sec = propeller_tip_mach * operating_atm.speed_of_sound() / (propeller_diameter / 2)
propeller_rpm = propeller_rads_per_sec * 30 / np.pi
motor_kv = propeller_rpm / battery_voltage

thrust_climb = togw_design * g * np.sind(min_climb_angle) + aero["D"]



### POWER
for i in range(N-1):
    solar_flux = solar_flux_profile[i]  # W / m^2 (numeric constant)
    solar_area = solar_panels_n * solar_panel_side_length**2 # m^2
    power_generated = solar_flux * solar_area * solar_cell_efficiency / energy_generation_margin
    power_used = (power_shaft_cruise + 8)  # 8w to run avionics
    net_energy = (power_generated - power_used) * (dt / 3600)  # Wh

    battery_update = np.softmin(battery_states[i] + net_energy, battery_capacity, hardness=10)
    opti.subject_to(battery_states[i+1] == battery_update)



### Mass
# Power
mass_solar_cells = 0.008 * solar_panels_n
mass_batteries = propulsion_electric.mass_battery_pack(
    battery_capacity_Wh=battery_capacity,
    battery_pack_cell_fraction=0.95
)
num_packs = battery_capacity / pack_energy_Wh
mass_wires = propulsion_electric.mass_wires(
    wire_length=wingspan / 2,
    max_current=power_out_max / battery_voltage,
    allowable_voltage_drop=battery_voltage * 0.01,
    material="aluminum"
)

# Avionics
mass_speedybee = .055
mass_gps = 0.012
mass_telemtry = 0.026
mass_receiver = 0.018
mass_power_board = 0.075 * 2 # 75g estimate for each power board
mass_avionics = mass_speedybee + mass_gps + mass_telemtry + mass_receiver + mass_power_board

# Actuators
mass_servos = .02 * 4 # 20g a servo

# Propulsion
mass_motor_raw = propulsion_electric.mass_motor_electric(
    max_power= power_out_max / propeller_n,
    kv_rpm_volt=motor_kv,
    voltage=battery_voltage
) * propeller_n
mass_motors_mounted = mass_motor_raw * 2 # marign to account for motor mounts.
mass_esc = propeller_n * propulsion_electric.mass_ESC(max_power=power_out_max)
mass_propellers = propeller_n * propulsion_propeller.mass_hpa_propeller(
    diameter=propeller_diameter, 
    max_power=power_out_max
)

# Structures
mass_hstab = hor_stabilizer.area("wetted")*1.2 * 0.6 # 600g per meter squared of wetted area
mass_vstab = (vert_stabilizer_L.area("wetted")*1.2 + vert_stabilizer_R.area("wetted")) * 0.6 # 600g per meter squared of wetted area
mass_main_wing =  main_wing.area("wetted")*1.5 * 0.700 # 770g per meter squared of planform area
mass_boom = 2 * 0.09 * boom_length  # kg (90 g/m per boom)
mass_fuselages = 0.2 # rough estimate


## Total
total_mass = (
    mass_solar_cells + 
    mass_batteries + 
    mass_wires + 
    mass_avionics +
    mass_servos +
    mass_motors_mounted + 
    mass_esc + 
    mass_propellers +
    mass_main_wing + mass_hstab + mass_vstab +
    mass_boom +
    mass_fuselages
)



### CONSTRAINTS
# Mission
opti.subject_to(total_mass < togw_design)

# Performance
opti.subject_to(thrust_cruise >= aero["D"])
opti.subject_to(power_out_max >= power_shaft_cruise)
opti.subject_to(power_out_max >= thrust_climb * airspeed / eta_prop)
opti.subject_to(motor_kv >= 150)

# Aerodynamic
opti.subject_to(aero["L"] >= togw_design * g)
opti.subject_to(aero["Cm"] == 0)
opti.subject_to(aero["CL"] <= CLmax_cruise)

# Stall-speed margin (equivalent to constraining required CL below CLmax at cruise)
V_stall = np.sqrt(2 * (togw_design * g) / (operating_atm.density() * S_w * CLmax_cruise))
opti.subject_to(airspeed >= stall_speed_margin * V_stall)
opti.subject_to(chordlen >= solar_panels_n_rows * 0.13 + 0.05) # ! Justify this addition of 0.1
opti.subject_to(wing_airfoil.max_thickness() * chordlen >= 0.025)  # must accomodate batteries (20mm)
opti.subject_to(wingspan >= 0.13 * solar_panels_n / solar_panels_n_rows)  # Must be able to fit all of our solar panels 13cm each
opti.subject_to(hstab_chordlen >= 0.125) # 12.5cm to make room for elevator & manufacturability
opti.subject_to(vstab_root_chord >= hstab_chordlen)  # enforce an actual taper (root >= tip) and avoid negative root chord

# Stability
opti.subject_to(L_H >= 1e-3)
opti.subject_to(L_V >= 1e-3)
opti.subject_to(boom_length >= chordlen)
opti.subject_to(static_margin >= SM_min)
opti.subject_to(static_margin <= SM_max)

# Power
opti.subject_to(battery_states >= battery_capacity * (1-allowable_battery_depth_of_discharge))
opti.subject_to(battery_states <= battery_capacity)
opti.subject_to(battery_states[0] <= battery_states[N-1])


### SOLVE
opti.minimize(total_mass)

try:
    sol = opti.solve()
    opti.save_solution()
except Exception:
    sol = opti.debug

def s(x):
    """
    Safely evaluates an expression at the current solution/debug iterate.
    (If the NLP never solved / never iterated, returns NaN instead of crashing.)
    """
    try:
        return sol.value(x)
    except Exception:
        return float("nan")

print("--- Environment ---")
print("Temperature (K):", s(operating_atm.temperature()))
print("Temperature (°F):", s(operating_atm.temperature() * 1.8 - 459.67))
print("Pressure:", s(operating_atm.pressure()))
print("Density:", s(operating_atm.density()))
print("Speed of sound:", s(operating_atm.speed_of_sound()))

print("\n---Performance---")
print("Airspeed (m/s):", s(airspeed))
print("Thrust cruise (N):", s(thrust_cruise))
print("Power out max (W):", s(power_out_max))
print("Mass (kg):", s(total_mass))
print("TOGW Design (kg):", s(togw_design))
print("Weight (N):", s(togw_design * g))

print("\n--- Aerodynamics ---")
print("CL", s(aero["CL"]))
print("CD", s(aero["CD"]))
print("L/D", s(aero["CL"]/aero["CD"]))
print("Total lift (N):", s(aero["L"]))
print("Total drag (N):", s(aero["D"]))
# print("Alpha cruise (°):", s(alpha_cruise))
print("CL cruise:", s(aero["CL"]))
print("Stall speed (m/s):", s(V_stall))

print("\n--- Stability ---")
print("Static margin:", s(static_margin))
print("V_H (actual):", s((hstab_area * L_H) / (S_w * MAC_w)))
print("V_V (actual):", s((vstab_area_total * L_V) / (S_w * b_w)))

print("\n--- Wing Dimensions ---")
print("Main wing area (m^2):", s(main_wing.area()))
print("Main wing AR:", s(main_wing.aspect_ratio()))
print("Main wing span (m):", s(wingspan))
print("Main wing chord (m):", s(chordlen))
print("Main wing, struct defined AoA (°): ", s(struct_defined_aoa))
print("Hstab AoA (°):", s(hstab_aoa))
print("Hstab area (m^2):", s(hstab_area))
print("Hstab AR:", s(hstab_AR))
print("Hstab span (m):", s(hstab_span))
print("Hstab chord (m):", s(hstab_chordlen))
print("Vstab area (each) (m^2):", s(vstab_area))
print("Vstab area (total) (m^2):", s(vstab_area_total))
print("Vstab AR (each):", vstab_AR)
print("Vstab span (m):", s(vstab_span))
print("Vstab root chord (m):", s(vstab_root_chord))
print("Vstab tip chord (matches hstab) (m):", s(hstab_chordlen))

print("\n--- Structral Dimensions ---")
print("cg_le_dist:", s(cg_le_dist))
print("Inter-boom spacing (m):", s(boom_y)*2)
print("Main wing mass:", s(mass_main_wing))
print("Hstab mass (kg):", s(mass_hstab))
print("Vstab mass (both) (kg):", s(mass_vstab))
print("Boom length (m):", s(boom_length))
print("Boom mass (both) (kg):", s(mass_boom))
print("Fuselage mass (both) (kg):", s(mass_fuselages))

print("\n--- Power ---")
print("Solar cells #:", s(solar_panels_n))
print("Solar cell mass (kg):", s(mass_solar_cells))
print("Battery capacity (Wh):", s(battery_capacity))
print("Battery mass (kg):", s(mass_batteries))
print("Battery pack (#):", s(num_packs))
print("Wire mass (kg):", s(mass_wires))

print("\n--- Avionics ---")
print("Avionics mass (kg):", mass_avionics)
print("Power board mass (kg):", s(mass_power_board))
print("Telemtry mass (kg):", s(mass_telemtry))
print("Receiver mass (kg):", s(mass_receiver))
print("Wiring mass (kg):", s(mass_wires))
print("Servo mass (kg):", s(mass_servos))
print("Avionics mass (total) (kg):", s(mass_avionics))

print("\n--- Propulsion ---")
print("Motor #:", s(propeller_n))
print("Motor RPM:", s(propeller_rpm))
print("Motor KV", s(motor_kv))
print("Propeller diameter:", s(propeller_diameter))
print("Propeller mass:", s(mass_propellers))
print("Motors mass:", s(mass_motor_raw))
print("ESCs mass:", s(mass_esc))



# for k, v in aero.items():
#     print(f"{k.rjust(4)} : {sol(aero[k])}")

# vlm=sol(vlm)
# vlm.draw()


airplane_sol = None
try:
    if not callable(sol):
        raise TypeError(f"Solution object is not callable (type: {type(sol)}).")
    airplane_sol = sol(airplane)
    airplane_sol.draw()
except Exception as e:
    print(f"[draw] Skipping airplane draw (no solved solution): {e}")

# Export to XFLR5 (XML)
# Note: AeroSandbox's XFLR5 exporter supports exactly 3 lifting surfaces: main wing, elevator, and fin.
# Our geometry has a twin-fin (two vertical stabilizers), so we synthesize a single centerline fin for export.
xflr5_filename = "output/rev6_xflr5.xml"
try:
    if airplane_sol is None:
        raise RuntimeError("No solved airplane geometry available to export.")

    # Pick main wing and elevator directly from solved airplane.
    mainwing_xflr = airplane_sol.wings[0]
    elevator_xflr = airplane_sol.wings[1]

    # Build a single centerline fin using solved values (based on the per-fin geometry).
    vstab_root_chord_xflr = float(s(vstab_root_chord))
    vstab_tip_chord_xflr = float(s(hstab_chordlen))
    vstab_span_xflr = float(s(vstab_span))
    boom_length_xflr = float(s(boom_length))

    fin_xflr = asb.Wing(
        name="Vertical Stabilizer (XFLR5)",
        symmetric=False,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0.0, 0.0, 0.0],
                chord=vstab_root_chord_xflr,
                twist=0.0,
                airfoil=tail_airfoil,
            ),
            asb.WingXSec(
                xyz_le=[vstab_root_chord_xflr / 4, 0.0, vstab_span_xflr],
                chord=vstab_tip_chord_xflr,
                twist=0.0,
                airfoil=tail_airfoil,
            ),
        ],
    ).translate([boom_length_xflr, 0.0, 0.0])

    airplane_sol.export_XFLR5_xml(
        xflr5_filename,
        include_fuselages=False,  # not implemented in AeroSandbox exporter
        mainwing=mainwing_xflr,
        elevator=elevator_xflr,
        fin=fin_xflr,
    )
    print(f"[XFLR5] Exported: {xflr5_filename}")
except Exception as e:
    print(f"[XFLR5] Skipping export ({xflr5_filename}): {e}")

# Export to CAD (STEP) via CadQuery geometry
# step_filename = "output/rev6.step"
# try:
#     airplane_sol.export_cadquery_geometry(
#         step_filename,
#         minimum_airfoil_TE_thickness=0.001,
#     )
#     print(f"[CAD] Exported STEP: {step_filename}")
# except Exception as e:
#     print(f"[CAD] STEP export failed ({step_filename}): {e}")

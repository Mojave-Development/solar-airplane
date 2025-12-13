import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.library import power_solar, propulsion_electric, propulsion_propeller
from aerosandbox.atmosphere import Atmosphere
import numpy as onp


opti=asb.Opti(cache_filename="output/soln1.json")


### CONSTANTS
## Mission
mission_date = 100
operating_lat = 37.398928
togw_max = 12 # kg
temperature_high = 278 # in Kelvin --> this is 60 deg F addition to ISA temperature at 0 meter MSL
operating_altitude = 1200 # in meters
operating_atm = Atmosphere(operating_altitude, temperature_deviation=temperature_high)

## Aerodynamics
# Airfoils
wing_airfoil = asb.Airfoil("s4110")
tail_airfoil = asb.Airfoil("naca0010")

# Main wing
polyhedral_angle = 10

# Tail sizing (via Tail Volume Coefficients)
V_H = 0.50  # Horizontal tail volume coefficient
V_V = 0.020  # Vertical tail volume coefficient

# Static margin constraints (dimensionless, SM = (x_np - x_cg) / MAC)
SM_min = 0.05
SM_max = 0.6

## Structural
structural_mass_markup = 1.2

## Power
battery_voltage = 22.2
N = 180  # Number of discretization points
time = onp.linspace(0, 24 * 60 * 60, N)  # s (numeric; does not need to be symbolic)
dt = onp.diff(time)[0]  # s
solar_panels_n_rows = 2
solar_encapsulation_eff_hit = 0.1 # Estimated 10% efficieincy loss from encapsulation.
solar_cell_efficiency = 0.243 * (1 - solar_encapsulation_eff_hit)
energy_generation_margin = 1.05
allowable_battery_depth_of_discharge = 0.85  # How much of the battery can you actually use?

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

# Empennage (sized by tail volume coeffs)
hstab_AR = opti.variable(init_guess=6.0, lower_bound=2.0, upper_bound=20.0, scale=5, category="hstab_AR")
vstab_AR = 2.0
hstab_aoa = opti.variable(init_guess=-5, lower_bound=-5, upper_bound=0, scale=5, category="hstab_aoa")

# Structural
boom_length = opti.variable(init_guess=0.5, lower_bound=0.1, upper_bound=1, scale=0.5, category="boom_length")
boom_spacing_frac = 0.30
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
            ],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=chordlen,
            twist=struct_defined_aoa,  # degrees
            airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
        ),
        asb.WingXSec(  # Mid
            xyz_le=[0.00, boom_y, 0],
            chord=chordlen,
            twist=struct_defined_aoa,
            airfoil=wing_airfoil,
        ),
        asb.WingXSec(  # Tip
            xyz_le=[0.00, wingspan / 2, np.sin(10 * np.pi / 180) * 0.5 * wingspan / 2],
            chord=0.125,
            twist=struct_defined_aoa,
            airfoil=wing_airfoil,
        ),
    ],
)

# Tail sizing from volume coefficients + aspect ratio (rectangular planforms)
S_w = main_wing.area()
MAC_w = main_wing.mean_aerodynamic_chord()
b_w = main_wing.span()

# Vertical tail
L_V = boom_length - cg_le_dist
vstab_area_total = V_V * S_w * b_w / L_V
vstab_area = 0.5 * vstab_area_total
vstab_span = np.sqrt(vstab_AR * vstab_area)  # "span" here is height
vstab_chordlen = np.sqrt(vstab_area / vstab_AR)
x_le_vstab = boom_length - 0.25 * vstab_chordlen

vert_stabilizer_L = asb.Wing(
    name="Vertical Stabilizer L",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=vstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
        asb.WingXSec(
            xyz_le=[0.00, 0, vstab_span],
            chord=vstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
    ],
).translate([x_le_vstab, boom_y, 0])

vert_stabilizer_R = asb.Wing(
    name="Vertical Stabilizer R",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=vstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
        asb.WingXSec(
            xyz_le=[0.00, 0, vstab_span],
            chord=vstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
    ],
).translate([x_le_vstab, -boom_y, 0])

# Horizontal tail
hstab_ac_x = boom_length
L_H = hstab_ac_x - cg_le_dist
hstab_area = V_H * S_w * MAC_w / L_H
hstab_span = boom_y * 2
hstab_chordlen = hstab_area / hstab_span
x_le_hstab = boom_length - 0.25 * hstab_chordlen

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
).translate([x_le_hstab, 0, vstab_span])

# Booms
boom_radius = 0.01
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


airplane = asb.Airplane(
    name="rev 6",
    xyz_ref=[cg_le_dist, 0, 0],  # CG location (measured from main wing LE)
    wings=[main_wing, hor_stabilizer, vert_stabilizer_L, vert_stabilizer_R],
    fuselages=[left_pod, right_pod, boom_L, boom_R],
)


### AERODYNAMICS
vlm = asb.AeroBuildup(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        atmosphere=operating_atm, # FIX!
        velocity=airspeed,  # m/s
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
propeller_rads_per_sec = propeller_tip_mach * Atmosphere(altitude=1100).speed_of_sound() / (propeller_diameter / 2)
propeller_rpm = propeller_rads_per_sec * 30 / np.pi
motor_kv = propeller_rpm / battery_voltage

thrust_climb = togw_design * 9.81 * np.sind(30) + aero["D"]


### POWER
for i in range(N-1):
    solar_flux = solar_flux_profile[i]  # W / m^2 (numeric constant)

    solar_area = solar_panels_n * 0.125**2 # m^2
    power_generated = solar_flux * solar_area * solar_cell_efficiency / energy_generation_margin
    power_used = (power_shaft_cruise + 8)  # 8w to run avionics
    net_energy = (power_generated - power_used) * (dt / 3600)  # Wh

    battery_update = np.softmin(battery_states[i] + net_energy, battery_capacity, hardness=10)
    opti.subject_to(battery_states[i+1] == battery_update)


### Mass
# Power
mass_solar_cells = 0.01 * solar_panels_n
mass_batteries = propulsion_electric.mass_battery_pack(
    battery_capacity_Wh=battery_capacity,
    battery_pack_cell_fraction=0.95
)
num_packs = battery_capacity / (5 * 6 * 3.7) # 5 ah, 6 cells, 3.7 V/cell
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
mass_motors_mounted = mass_motor_raw * 2
mass_esc = propeller_n * propulsion_electric.mass_ESC(max_power=power_out_max)
mass_propellers = propeller_n * propulsion_propeller.mass_hpa_propeller(
    diameter=propeller_diameter, 
    max_power=power_out_max
)

# Structures
mass_hstab = hor_stabilizer.area() * 0.5
mass_vstab = (vert_stabilizer_L.area() + vert_stabilizer_R.area()) * 0.5
mass_main_wing = main_wing.area() * 0.7
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
opti.subject_to(power_out_max >= thrust_climb * airspeed)
opti.subject_to(motor_kv >= 150)

# Aerodynamic
opti.subject_to(aero["L"] >= togw_design * 9.81)
opti.subject_to(chordlen >= solar_panels_n_rows * 0.13 + 0.05) # ! Justify this addition of 0.1
opti.subject_to(wing_airfoil.max_thickness() * chordlen >= 0.025)  # must accomodate batteries (20mm)
opti.subject_to(wingspan >= 0.13 * solar_panels_n / solar_panels_n_rows)  # Must be able to fit all of our solar panels 13cm each
opti.subject_to(hstab_chordlen >= 0.2) # Fit 1 row of panels w/ room for elevator

# Stability
tail_gap = 0.15
opti.subject_to(L_H >= 1e-3)
opti.subject_to(L_V >= 1e-3)
opti.subject_to(x_le_hstab >= chordlen + tail_gap)
opti.subject_to(static_margin >= SM_min)
opti.subject_to(static_margin <= SM_max)

# Power
opti.subject_to(battery_states >= battery_capacity * (1-allowable_battery_depth_of_discharge))
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

print("---Performance---")
print("Airspeed:", s(airspeed))
print("Thrust cruise:", s(thrust_cruise))
print("Power out max:", s(power_out_max))
print("Mass:", s(total_mass))
print("TOGW Max:", s(togw_max))
print("TOGW Design:", s(togw_design))
print("Weight:", s(togw_design * 9.81))

print("\n--- Aerodynamics ---")
print("CL", s(aero["CL"]))
print("CD", s(aero["CD"]))
print("L/D", s(aero["CL"]/aero["CD"]))
print("Total lift:", s(aero["L"]))
print("Total drag:", s(aero["D"]))

print("\n--- Wing Dimensions ---")
print("Main wing span:", s(wingspan))
print("Main wing chord length:", s(chordlen))
print("Main wing AR:", s(main_wing.aspect_ratio()))
print("Struct defined AoA: ", s(struct_defined_aoa))
print("Hstab AoA:", s(hstab_aoa))
print("Hstab AR:", hstab_AR)
print("Hstab area:", s(hstab_area))
print("Hstab span:", s(hstab_span))
print("Hstab chord:", s(hstab_chordlen))
print("Vstab AR:", vstab_AR)
print("Vstab area (each):", s(vstab_area))
print("Vstab area (total):", s(vstab_area_total))
print("Vstab span:", s(vstab_span))
print("Vstab chord:", s(vstab_chordlen))
print("Static margin:", s(static_margin))
print("V_H (actual):", s((hstab_area * L_H) / (S_w * MAC_w)))
print("V_V (actual):", s((vstab_area_total * L_V) / (S_w * b_w)))

print("\n--- Structral Dimensions ---")
print("cg_le_dist:", s(cg_le_dist))
print("Boom length:", s(boom_length))
print("Boom mass:", s(mass_boom))
print("Hstab mass:", s(mass_hstab))
print("Vstab mass:", s(mass_vstab))
print("Main wing mass:", s(mass_main_wing))
print("Fuselage mass:", s(mass_fuselages))

print("\n--- Power ---")
print("Solar cells #:", s(solar_panels_n))
print("Solar cell mass:", s(mass_solar_cells))
print("Battery capacity:", s(battery_capacity))
print("Battery mass:", s(mass_batteries))
print("Battery pack #:", s(num_packs))
print("Wire mass:", s(mass_wires))

print("\n--- Power ---")
print("Avionics mass:", mass_avionics)

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


airplane_sol = sol(airplane)
airplane_sol.draw()
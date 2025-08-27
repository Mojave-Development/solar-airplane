import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.library import power_solar
from aerosandbox.atmosphere import Atmosphere
from lib.aero import calculate_skin_friction
import os


opti=asb.Opti(cache_filename=os.path.join(os.getcwd(),"output","soln1.json"))


# ---------- CONSTANTS ----------
# --- Mission ---
mission_date = 100
lat = 37.398928

temperature_high = 278 # in Kelvin --> this is 60 deg F addition to ISA temperature at 0 meter MSL
operating_altitude = 1200 # in meters
operating_atm = Atmosphere(operating_altitude, temperature_deviation=temperature_high)


# --- Aerodynamic ---
# Airfoils
wing_airfoil = asb.Airfoil("s1210")
tail_airfoil = asb.Airfoil("naca0010")

# Main wing
polyhedral_angle = 10

# Vstab dimensions
vstab_span = 0.3
vstab_chordlen = 0.15

# --- Power ---
N = 100  # Number of discretization points
time = np.linspace(0, 24 * 60 * 60, N)  # s
dt = np.diff(time)[0]  # s
solar_cell_efficiency = 0.243 * 0.9 # 24.3% efficient


# ---------- VARIABLES ----------
# --- Aerodynamic ---
airspeed = opti.variable(init_guess=15, lower_bound=5, upper_bound=30, scale=5, category="airspeed")

# Main wing
wingspan = opti.variable(init_guess=3.4, lower_bound=2, upper_bound=7, scale=2, category="wingspan")
chordlen = opti.variable(init_guess=0.3, scale=1, category="chordlen")
struct_defined_aoa = opti.variable(init_guess=2, lower_bound=0, upper_bound=2, scale=1, category="struct_aoa")
cg_le_dist = opti.variable(init_guess=0.05, lower_bound=0, scale=1, category="cg_le_dist")

# Hstab
hstab_span = opti.variable(init_guess=0.5, lower_bound=0.3*wingspan, upper_bound=0.75*wingspan, scale=1, category="hstab_span")
hstab_chordlen = opti.variable(init_guess=0.2, lower_bound=0.15, upper_bound=0.4, scale=0.5, category="hstab_chordlen")
hstab_aoa = opti.variable(init_guess=-5, lower_bound=-5, upper_bound=5, category="hstab_aoa")

# Body
boom_length = opti.variable(init_guess=2, lower_bound=0.3*wingspan, upper_bound=1*wingspan, scale=1, category="boom_length")

# --- Power ---
n_solar_panels = opti.variable(init_guess=40, lower_bound=10, category="power", scale=10)
battery_cap = opti.variable(init_guess=1500, lower_bound=100, category="power", scale=1000)  # initial battery energy in Wh
battery_states = opti.variable(n_vars=N, init_guess=500, category="power", scale=100)


# ---------- GEOMETRIES ----------
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
            xyz_le=[0.00, 0.5 * wingspan / 2, 0],
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
).translate([boom_length, 0, 0])

vert_stabilizer_left = asb.Wing(
    name="Vertical Stabilizer",
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
).translate([boom_length + hstab_chordlen, -hstab_span/2, 0])

vert_stabilizer_right = asb.Wing(
    name="Vertical Stabilizer",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=vstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
        asb.WingXSec(
            xyz_le=[0, 0, vstab_span],
            chord=vstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
    ],
).translate([boom_length + hstab_chordlen, hstab_span/2, 0])

left_pod = asb.Fuselage(  # left pod fuselage
    name="Fuselage",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0.2 * xi, -hstab_span/2, -0.02],
            radius=0.4
            * asb.Airfoil("dae51").local_thickness(
                x_over_c=xi
            ),  # half a meter fuselage. Starting at LE and 0.5m forward
        )
        for xi in np.cosspace(0, 1, 30)
    ],
)

right_pod = asb.Fuselage(  # right pod fuselage
    name="Fuselage",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0.2 * xi, hstab_span/2, -0.02],
            radius=0.4
            * asb.Airfoil("dae51").local_thickness(
                x_over_c=xi
            ),  # half a meter fuselage. Starting at LE and 0.5m forward
        )
        for xi in np.cosspace(0, 1, 30)
    ],
)

### Define the 3D geometry you want to analyze/optimize.
# Here, all distances are in meters and all angles are in degrees.
airplane = asb.Airplane(
    name="rev 6",
    xyz_ref=[0.1 * chordlen, 0, 0],  # CG location
    wings=[main_wing, hor_stabilizer, vert_stabilizer_left, vert_stabilizer_right],
    fuselages=[ left_pod, right_pod],
)


# ---------- AERODYNAMICS ----------
vlm = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        atmosphere=operating_atm, # FIX!
        velocity=airspeed,  # m/s
    ),
)
aero = vlm.run_with_stability_derivatives()  # Returns a dictionary

# VLM does not calcualte parasitic drag, we must add this manually
CD0 = (
    calculate_skin_friction(chordlen, airspeed) * main_wing.area(type="wetted") / main_wing.area()
    + calculate_skin_friction(hstab_chordlen, airspeed) * hor_stabilizer.area(type="wetted") / main_wing.area()
    + 2* calculate_skin_friction(vstab_chordlen, airspeed) * vert_stabilizer_left.area(type="wetted") / main_wing.area()
    + 2 * calculate_skin_friction(0.2, airspeed) * left_pod.area_wetted() / main_wing.area()
)

drag_parasite = 0.5 * 1.29 * airspeed**2 * main_wing.area() * CD0

aero["CD_tot"] = aero["CD"] + CD0
aero["D_tot"] = aero["D"] + drag_parasite
aero["power"] = aero["D_tot"] * airspeed


# ---------- POWER ----------
for i in range(N-1):
    solar_flux = power_solar.solar_flux(
        latitude=lat,
        day_of_year=mission_date,
        time=time[i],
        altitude=operating_altitude,
        panel_azimuth_angle=0,
        panel_tilt_angle=0
    ) # W / m^2

    solar_area = n_solar_panels * 0.125**2 # m^2
    power_generated = solar_flux * solar_area * solar_cell_efficiency
    power_used = (aero["power"] + 8) * 1.2  # 8w to run avionics
    net_energy = (power_generated - power_used) * (dt / 3600)  # Wh

    battery_update = np.softmin(battery_states[i] + net_energy, battery_cap, hardness=10)
    opti.subject_to(battery_states[i+1] == battery_update)


# ---------- WEIGHT ----------
# --- Power ---
solar_cell_mass = 0.015 * n_solar_panels
num_packs = battery_cap / (5 * 6 * 3.7) # 5 ah, 6 cells, 3.7 V/cell
battery_mass = num_packs * 0.450

# --- Structures ---
foam_volume = main_wing.volume() + hor_stabilizer.volume() + 2*vert_stabilizer_left.volume()
foam_mass = foam_volume * 30.0  # foam 30kg.m^2
spar_mass = (wingspan / 2 + boom_length) * 0.09  # 90g/m carbon spar 22mm
fuselages_mass = 1.0  # 1kg for all fuselage pods

# --- Total ---
weight = 9.81 * (
    solar_cell_mass + 
    battery_mass + 
    foam_mass + 
    spar_mass + 
    fuselages_mass
)

# ---------- STABILITY ----------
static_margin = (cg_le_dist - aero["x_np"]) / np.softmax(1e-6, main_wing.mean_aerodynamic_chord(), hardness=10)


# ---------- CONSTRAINTS ----------
# --- Mission
opti.subject_to(weight <= 7 * 9.81)
# --- Aerodynamic ---
opti.subject_to(aero["L"] == weight)
opti.subject_to(wing_airfoil.max_thickness() * chordlen > 0.030)  # must accomodate main spar (22mm)
opti.subject_to(wingspan > 0.13 * n_solar_panels)  # Must be able to fit all of our solar panels 13cm each

# --- Stability ---
opti.subject_to(cg_le_dist <= 0.25 * chordlen)
# opti.subject_to(static_margin > 0.1)
# opti.subject_to(static_margin < 0.5)

# --- Power ---
opti.subject_to(battery_states > 50) #Sorry for splitting these up. Windows quirk - CRJ 26AUG2025
opti.subject_to(battery_states[0] == battery_cap)
opti.subject_to(battery_states[N-1] == battery_cap)


# ---------- SOLVE ----------
opti.minimize(wingspan)
sol = opti.solve()
opti.save_solution()

print("Airspeed:", sol(airspeed))
print("Wingspan:", sol(wingspan))
print("Chordlen:", sol(chordlen))
print("Hstab AoA:", sol(hstab_aoa))
print("Hstab span:", sol(hstab_span))
print("Solar cell mass:", sol(solar_cell_mass))
print("Solar cells #:", sol(n_solar_panels))
print("Battery mass:", sol(battery_mass))
print("Battery pack #:", sol(num_packs))
print("Foam mass:", sol(foam_mass))
print("Spar mass:", sol(spar_mass))
print("Fuselages mass:", sol(fuselages_mass))

for k, v in aero.items():
    print(f"{k.rjust(4)} : {sol(aero[k])}")

vlm=sol(vlm)
vlm.draw()

airplane=sol(airplane)
airplane.draw()
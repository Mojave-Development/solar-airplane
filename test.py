from aerosandbox.library import propulsion_electric

check = propulsion_electric.motor_electric_performance(
    23,
    rpm=4000,
    kv=200
)

print(check)
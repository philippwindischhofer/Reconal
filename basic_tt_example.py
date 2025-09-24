import defs, numpy as np
from propagation import TravelTimeCalculator

# Set up domain.
rmax = 1000 # rmin always assumed to be 0
zmin, zmax = -1000, 1000
dr, dz = 1, 1   # Grid spacing in r, z directions
npts_r, npts_z = int(rmax / dr) + 1, int((zmax - zmin) / dr) + 1    # Number of grid nodes (note +1)
tx_pos = np.array([0, -60])  # Source position

# In this example, we use one of the built in Reconal ice models.
# See full_tt_example.ipynb for an example with a NuRadioMC IceModel() object.
ior, grad_ior = defs.ior_exp1, defs.grad_ior_exp1
reflection_at_z = 0.0

# Solve!
nrays = 50  # Number of big rays to generate refracted map. To skip refracted map, set to 0.
ttc = TravelTimeCalculator(tx_pos[1], zmin, zmax, rmax, npts_z, npts_r)
ttc.set_ior_and_solve(ior, grad_ior, nrays, reflection_at_z)

# Extract traveltime maps as ndarrays.
for comp, field in ttc.travel_time_fields.items():
    np.save(f'{comp} travel time map.npy', field.values)

# Use traveltime maps to find details about a specific point.
coord = np.array([530.0, -600.0, 0.0]) # Third coordinate should always be 0
traveltimes = {}
launch_vecs = {}

for comp, field in ttc.travel_time_fields.items():
    tt = field.value(coord) # Traveltime determined through trilinear interpolation
    if np.isfinite(tt):
        traveltimes[comp] = tt
        grad = field.gradient.value(coord)[:-1] # Launch vector determined through traveltime gradient
        launch_vecs[comp] = grad / np.linalg.norm(grad) # Retrieve unit vector

print(f"Starting point: {tx_pos} \nEnding point: {coord[:-1]} \nTraveltimes: {traveltimes} \nLaunch vectors: {launch_vecs}")

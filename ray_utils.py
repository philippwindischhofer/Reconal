import numpy as np
from scipy.constants import c

speed_of_light = c / 1e9

def get_theta_min(src, ior, reflection_at_z):
    """
    Returns angle (in degrees) such that turnover occurs at reflection depth.
    """
    if ior(reflection_at_z) <= ior(src[1]):
        theta = np.arcsin(ior(reflection_at_z) / ior(src[1])) # Snell's law
        return np.degrees(theta)
    else:
        return 0
    
def get_adaptive_dr(max_theta, ior, grad_ior, z, step):
    """
    Adjusts radial raytracing step based on percent change in index of refraction.
    """
    grad = grad_ior(z)
    if grad == 0.0:
        return step
    else:
        dr = abs(max_theta * ior(z) / grad)
        return min(dr, step)

def ray_tracer(theta, src, ior, grad_ior, rmax, z_min, z_max, step, max_theta):
    """
    Adaptive Snell's law raytracer for plane-stratified media.
    """
    ray = np.full((3, int(rmax / max_theta)), np.nan, dtype = float)
    ray[:, 0] = src + [0,]
    
    theta = np.radians(theta)

    if theta > np.pi / 2:
        theta += np.pi

    ior_old = ior(src[1])

    dr = get_adaptive_dr(max_theta, ior, grad_ior, src[1], step)
    turnover = None

    for i in range(int(rmax / max_theta) + 1):
        if ray[0, i] > rmax or ray[1, i] < z_min or ray[1, i] > z_max:
            break
        
        ray[0, i + 1] = ray[0, i] + dr  # Update r value (range)
        ray[1, i + 1] = ray[1, i] + dr / np.tan(theta)  # Update z value (depth)
        ray[2, i + 1] = ray[2, i] + np.abs(dr / np.sin(theta)) * ior_old / speed_of_light # Update traveltime (dt = ds * n / c)
        
        ior_new = ior(ray[1, i + 1])
        
        if (ior_old / ior_new) * np.sin(theta) <= 1:
            theta = np.arcsin((ior_old / ior_new) * np.sin(theta)) # Snell's law
        else: # Ray reflects/refracts and begins to propagate downward
            turnover = ray[:,i] # Record point where ray changes direction
            theta = 2 * np.pi - theta
        
        dr = get_adaptive_dr(max_theta, ior, grad_ior, ray[1, i + 1], step)

        ior_old = ior_new

    return ray, turnover

def get_rays(src, ior, grad_ior, rmax, z_min, z_max, mesh, step = 1.0):
    """
    Loop to execute raytracer. Corrects for possible division by 0 in case of rays traveling straight up or down.
    """

    max_theta = 0.001

    rays = np.full((len(mesh), 3, int(rmax // max_theta) + 1), np.nan)
    turnover = np.full((len(mesh), 3), np.nan)
    
    for i, theta in enumerate(mesh):
        if 0 < theta < 180: # Ray-tracer requires horizontal propagation; filter out strictly vertical rays
            rays[i], turnover[i] = ray_tracer(theta, src, ior, grad_ior, rmax, z_min, z_max, step, max_theta)
        elif theta == 0.0: # ray goes straight up
            rays[i, 0] = src[0]
            rays[i, 1] = src[1] + np.arange(0, int(rmax / max_theta)) * step
            rays[i][:, rays[i, 1] > z_max] = np.nan
            rays[i, 2] = (rays[i, 1] - src[1]) * ior(rays[i, 1]) / speed_of_light
        elif theta == 180.0: # ray goes straight down
            rays[i, 0] = src[0]
            rays[i, 1] = src[1] - np.arange(0, int(rmax / max_theta)) * step
            rays[i][:, rays[i, 1] < z_min] = np.nan
            rays[i, 2] = (src[1] - rays[i, 1]) * ior(rays[i, 1]) / speed_of_light
        else:
            raise ValueError("Launch angle must be between 0 and 180 (inclusive)")

    return rays, turnover


def get_special_bounds(src, ior, grad_ior, rmax, z_min, z_max, reflection_at_z, step = 1):
    """
    Returns caustic (bound for direct and refracted maps), largest reflected ray (reflected map), and turnover line (direct map).
    """
    theta_min = get_theta_min(src, ior, reflection_at_z) + 0.01
    if theta_min < 89.999:
        mesh = np.linspace(theta_min, 89.999, int((89.999 - theta_min) / 0.5))
    else:   # Source at surface
        mesh = [89.999]
    rays, turnover = get_rays(src, ior, grad_ior, rmax, z_min, z_max, mesh, step)
    rvals = np.arange(0, rmax, step)

    # Generating caustic (direct map, big ray bounds)
    coords = np.copy(rays).swapaxes(1, 2) # Sort into coordinate pairs; leave out reflected ray
    coords = coords[~np.isnan(coords).any(axis = 2)] # Remove NaN values, combine rays into one set of coordinates
    coords = coords[coords[:, 0].argsort()] # Sort by rvals
    caustic = np.full((int((rmax + 2) / step), 3), np.nan)
    
    for i, r in enumerate(rvals): # Select top edge of ray family (with tolerance for different step sizes)
        mask = np.logical_and(coords[:, 0] < r + step, coords[:, 0] > r - step) 
        idx = np.argmax(coords[mask][:, 1]) # For r-window (r - step, r + step), find the largest z-value
        caustic[i] = coords[mask][idx]

    caustic = caustic[caustic[:, 0] >= turnover[0, 0]].swapaxes(0, 1) # Values left of intersection with surface are unphysical; reject

    # Generating largest reflected ray (reflected map)
    reflected_bounds = np.copy(rays)[0, :-1].swapaxes(0, 1)
    reflected_bounds = reflected_bounds[~np.isnan(reflected_bounds).any(axis = 1)]
    reflected_bounds = reflected_bounds[reflected_bounds[:, 0] >= turnover[0, 0]].swapaxes(0, 1)

    # Generating turnover line (direct map)
    turnover = np.concatenate((turnover, np.swapaxes(rays[-1], 0, 1)), axis = 0)
    turnover = turnover[~np.isnan(turnover).any(axis = 1)] # Remove NaN values 
    turnover = turnover[turnover[:, 1].argsort()].swapaxes(0, 1) # Sort by zvals

    return caustic, turnover, reflected_bounds
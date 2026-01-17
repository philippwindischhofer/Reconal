import numpy as np
from scipy.constants import c

def get_grazing_angle(src, ior, z_turnover):
    """
    Returns angle (in degrees) such that turnover occurs at reflection depth.
    """
    if ior(z_turnover) <= ior(src[1]):
        theta = np.arcsin(ior(z_turnover) / ior(src[1])) # Snell's law
        return np.degrees(theta)
    else:
        return 0
    
def get_adaptive_dr(max_dtheta, ior, grad_ior, z, step):
    """
    Adjusts radial raytracing step based on percent change in index of refraction.
    """
    grad = grad_ior(z)
    if grad == 0.0:
        return step
    else:
        dr = abs(max_dtheta * ior(z) / grad)
        return min(dr, step)

def ray_tracer(theta, src, ior, grad_ior, rmax, zmin, zmax, step, reflection_at_z = 0.0):
    """
    Adaptive Snell's law raytracer for plane-stratified media.
    """
    max_dtheta = 1e-4
    step_bound = min(step, np.abs(max_dtheta * ior(reflection_at_z) / grad_ior(reflection_at_z)))  # Lower bound for step size, to efficiently allocate memory. Assumes |dn/dz| takes on a maximum at z=0.
    ray = np.full((3, int(rmax / step_bound)), np.nan, dtype = float)
    ray[:, 0] = src + [0,]
    
    theta = np.radians(theta)

    if theta == np.pi / 2:
        theta -= 0.00001    # Correct so that a 90 degree ray will bend
    if theta > np.pi / 2:
        theta += np.pi

    ior_old = ior(src[1])

    dr = get_adaptive_dr(max_dtheta, ior, grad_ior, src[1], step)
    turnover = None

    for i in range(int(rmax / step_bound) - 2):

        if ray[0, i] > rmax or ray[1, i] < zmin or ray[1, i] > zmax:
            break
        
        ray[0, i + 1] = ray[0, i] + dr  # Update r value (range)
        ray[1, i + 1] = ray[1, i] + dr / np.tan(theta)  # Update z value (depth)
        ray[2, i + 1] = ray[2, i] + np.abs(dr / np.sin(theta)) * ior_old / (c / 1e9) # Update traveltime (dt = ds * n / c)

        ior_new = ior(ray[1, i + 1])
        
        if np.abs((ior_old / ior_new) * np.sin(theta)) <= 1:
            theta = np.arcsin((ior_old / ior_new) * np.sin(theta)) # Snell's law
        else: # Ray reflects/refracts and begins to propagate downward
            turnover = ray[:,i] # Record point where ray changes direction
            theta = 2 * np.pi - theta
        
        dr = get_adaptive_dr(max_dtheta, ior, grad_ior, ray[1, i + 1], step)

        ior_old = ior_new

    return ray[:, np.isfinite(ray[0])], turnover

def get_rays(src, ior, grad_ior, rmax, zmin, zmax, mesh, step = 1.0):
    """
    Loop to execute raytracer. Corrects for division by 0 in case of rays traveling straight up or down.
    """
    rays = []
    turnover = []
    
    for theta in mesh:
        if 0 < theta < 180: # Ray-tracer requires horizontal propagation; filter out strictly vertical rays
            
            output = ray_tracer(theta, src, ior, grad_ior, rmax, zmin, zmax, step)
            rays.append(output[0])
            turnover.append(output[1])

        elif theta == 0.0: # ray goes straight up
            
            ray = np.empty((3, int(rmax / step)))
            ray[0] = src[0]
            ray[1] = src[1] + np.arange(0, rmax, step)
            ray[:, ray[1] > zmax] = np.nan
            ray[2] = (ray[1] - src[1]) * ior(ray[1]) / (c / 1e9)
            
            rays.append(ray)
            turnover.append(None)

        elif theta == 180.0: # ray goes straight down

            ray = np.empty((3, int(rmax / step)))
            ray[0] = src[0]
            ray[1] = src[1] - np.arange(0, rmax, step)
            ray[:, ray[1] < zmin] = np.nan
            ray[2] = (src[1] - ray[1]) * ior(ray[1]) / (c / 1e9)

            rays.append(ray)
            turnover.append(None)

        else:
            raise ValueError("Launch angle must be between 0 and 180 (inclusive)")

    return rays, turnover

def get_edge(rays, turnover, rmax, step, which = 'upper'):
    rvals = np.arange(0, rmax, step)
    edge = np.full((len(rvals), 3), np.nan)

    if which == 'upper':
        choose = np.nanargmax
        bound = -np.inf
    elif which == 'lower':
        choose = np.nanargmin
        bound = np.inf
    
    rays_interp = np.empty((len(rays), 2, len(rvals)))
    for i, ray in enumerate(rays):
        rays_interp[i] = np.interp(rvals, ray[0], ray[1], left = bound, right = bound), np.interp(rvals, ray[0], ray[2], left = np.nan, right = np.nan)
    idxs = choose(rays_interp[:, 0], axis = 0)
    edge = rays_interp.swapaxes(1,2)[idxs, np.arange(len(rvals))]
    edge = np.concatenate((rvals[:, np.newaxis], edge), axis = 1)
    edge = edge[edge[:, 0] >= turnover[0][0]].swapaxes(0, 1)
    edge[np.isinf(edge)] = np.nan
    return edge

def get_special_bounds(src, ior, grad_ior, rmax, zmin, zmax, reflection_at_z, step = 1.0):
    """
    Returns caustic (bound for direct maps), largest reflected ray (reflected map), and turnover line (direct map).
    """
    
    theta_min = get_grazing_angle(src, ior, reflection_at_z) + 0.01
    if theta_min < 89.999:
        mesh = np.arange(theta_min, 89.999, 1)
    else:   # Source at surface
        mesh = [89.999]
    rays, turnover = get_rays(src, ior, grad_ior, rmax, zmin, zmax, mesh, step)

    if np.isnan(turnover[0]).all(): # Domain too small to include caustic
        return None, None

    # Generating caustic (direct map, big ray bounds)
    caustic = get_edge(rays, turnover, rmax, step)

    # Generating largest reflected ray (reflected map)
    reflected_bounds = rays[0][:-1].swapaxes(0, 1)
    reflected_bounds = reflected_bounds[reflected_bounds[:, 0] >= turnover[0][0]].swapaxes(0, 1)

    return caustic, reflected_bounds
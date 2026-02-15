import numpy as np
from scipy.constants import c

def get_grazing_angle(src, ior, z_turnover):
    """
    Returns angle (in degrees) such that turnover occurs at reflection depth.
    """
    z_turnover -= 1   # small numerical tolerance to account for fact that raytracer can overshoot
    if ior(z_turnover) <= ior(src[1]):
        theta = np.arcsin(ior(z_turnover) / ior(src[1])) # Snell's law
        return np.degrees(theta)
    else:
        return None
    
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
    rvals = np.arange(0, rmax + step, step)
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
    edge[edge[:, 0] < turnover[0][0]] = np.nan
    edge[np.isinf(edge)] = np.nan
    
    return edge.swapaxes(0, 1)

def get_caustic(src, ior, grad_ior, rmax, zmin, zmax, z_bounds = [0.0], step = 1.0):
    """
    Returns simple or swallowtail caustic (including shadow zone boundary).
    """

    if len(z_bounds) != 1 and len(z_bounds) != 3:
        raise ValueError("Coming soon?")
    
    caustic = {}    # Contains different sections of the caustic (SZB, pocket sides 1,2,3)
    critical_angles = [get_grazing_angle(src, ior, z) for z in z_bounds]
    critical_angles = [item for item in critical_angles if item is not None]
    critical_angles.append(90)
    tol = 0.01  # numerics...

    # The SZB is always part of the caustic; find this first
    ray_mesh = np.linspace(critical_angles[0], critical_angles[-1] - tol, 100)
    rays, turnover = get_rays(src, ior, grad_ior, rmax, zmin, zmax, ray_mesh, step)
    
    if np.isnan(turnover[0]).all(): # Domain too small to include caustic
        return None

    caustic[0] = get_edge(rays, turnover, rmax, step)
    
    if len(z_bounds) > 1 and src[1] < z_bounds[-1]: # 3-layer exponential produces a swallowtail caustic at certain depths.

        # start by finding boundary of "bottom bundle" of rays
        ray_mesh = np.linspace(critical_angles[-2] + tol, critical_angles[-1] - tol, 100)
        rays, turnover = get_rays(src, ior, grad_ior, rmax, zmin, zmax, ray_mesh, step)
        caustic[3] = get_edge(rays, turnover, rmax, step)

        # go a layer up. this is the problematic layer, so we throw more rays
        ray_mesh = np.linspace(critical_angles[1] + tol, critical_angles[2] - tol, 200)
        rays, turnover = get_rays(src, ior, grad_ior, rmax, zmin, zmax, ray_mesh, step)

        # find the problem rays (if any). these are rays that don't intersect the typical shadow zone boundary
        # and therefore must focus at the bottom of the pocket, forming the swallowtail shape
        mask = []
        for i, ray in enumerate(rays):
            ray_interp = np.array([caustic[0][0], np.interp(caustic[0][0], ray[0], ray[1])])
            dist_to_szb = np.abs(ray_interp[1] - caustic[0][1])
            if np.any(dist_to_szb < step):
                continue
            else:
                mask.append(i)
        problem_rays = [rays[i] for i in mask]   # lot of annoying list comprehension because ray arrays are not necessarily the same size
        if problem_rays:
            caustic[2] = get_edge(problem_rays, turnover, rmax, step, which = 'lower')
            caustic[2][:, caustic[2][1] - zmin < 5] = np.nan
            critical_angles.append(ray_mesh[mask[0]])
            critical_angles.sort()
        
        # remove the now-handled problem rays and get the last side of the swallowtail
        rays = [ray for i, ray in enumerate(rays) if i not in mask]
        caustic[1] = get_edge(rays, turnover, rmax, step)

        if problem_rays:
            diffs = np.abs(caustic[1][1] - caustic[2][1])
            caustic[1][:, np.nanargmin(diffs):] = np.nan
            caustic[2][:, np.nanargmin(diffs):] = np.nan

            diffs = np.abs(caustic[3][1] - caustic[2][1])
            caustic[3][:, :np.nanargmin(diffs)] = np.nan
            caustic[2][:, :np.nanargmin(diffs)] = np.nan

    return caustic, critical_angles

def get_reflected_bound(src, ior, grad_ior, rmax, zmin, zmax, reflection_at_z, step = 1.0):
    """
    Returns rightmost reflected ray.
    """
    theta_min = get_grazing_angle(src, ior, reflection_at_z) + 0.01
    
    if theta_min > 89.999:
        theta_min = 89.999
    
    ray, turnover = get_rays(src, ior, grad_ior, rmax, zmin, zmax, [theta_min], step)
    reflected_bound = ray[0][:-1].swapaxes(0, 1)
    reflected_bound = reflected_bound[reflected_bound[:, 0] >= turnover[0][0]].swapaxes(0, 1)

    return reflected_bound
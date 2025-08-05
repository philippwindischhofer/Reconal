import numpy as np
import matplotlib.pyplot as plt
import time

def get_theta_min(src, ior, reflection_at_z): # returns angle (in degrees) such that turnover occurs at reflection line
    if ior(reflection_at_z) / ior(src[1]) <= 1:
        theta = np.arcsin(ior(0) / ior(src[1])) # Snell's law
        return np.degrees(theta)
    else:
        return 0
    
def get_adaptive_dr(max_theta, ior, grad_ior, z, step): # adjusts r step based on % change in IOR
    dr = abs(max_theta * ior(z) / grad_ior(z))
    return min(dr, step)

def ray_tracer(theta, src, ior, grad_ior, rmax, z_range, step, max_theta): # adaptive ray tracer for plane-stratified media

    ray = np.full((2, int(rmax / max_theta)), np.nan, dtype = float)
    ray[:, 0] = src
    
    theta = np.radians(theta)

    if theta > np.pi / 2:
        theta += np.pi

    ior_old = ior(src[1])

    dr = get_adaptive_dr(max_theta, ior, grad_ior, src[1], step)
    turnover = None

    for i in range(int(rmax / max_theta) + 1):
        if ray[0, i] > rmax or ray[1, i] > z_range[1]:
            break
        
        ray[0, i + 1] = ray[0, i] + dr
        ray[1, i + 1] = ray[1, i] + dr / np.tan(theta)
        
        ior_new = ior(ray[1, i + 1])
        
        if (ior_old / ior_new) * np.sin(theta) <= 1:
            theta = np.arcsin((ior_old / ior_new) * np.sin(theta)) # Snell's law
        else: # Ray reflects/refracts and begins to propagate downward
            turnover = ray[:,i] # Record point where ray changes direction
            theta = 2 * np.pi - theta
        
        dr = get_adaptive_dr(max_theta, ior, grad_ior, ray[1, i + 1], step)

        ior_old = ior_new

    return ray, turnover

def get_rays(src, ior, grad_ior, rmax, z_range, mesh):

    step = 1
    max_theta = 0.001

    rays = np.full((len(mesh), 2, int(rmax // max_theta) + 1), np.nan)
    turnover = np.full((len(mesh), 2), np.nan)
    
    for i, theta in enumerate(mesh):
        if 0 < theta < 180: # Ray-tracer requires horizontal propagation; filter out strictly vertical rays
            rays[i], turnover[i] = ray_tracer(theta, src, ior, grad_ior, rmax, z_range, step, max_theta)
        elif theta == 0: # ray goes straight up
            rays[i, 0] = src[0]
            rays[i, 1] = src[1] + np.arange(0, int(rmax / max_theta)) * step
        elif theta == 180: # ray goes straight down
            rays[i, 0] = src[0]
            rays[i, 1] = src[1] - np.arange(0, int(rmax / max_theta)) * step
        else:
            raise ValueError("Launch angle must be between 0 and 180 (inclusive)")

    return rays, turnover

def get_caustic(src, ior, grad_ior, rmax, z_range, reflection_at_z):

    mesh = np.linspace(get_theta_min(src, ior, reflection_at_z), 89, 20)
    rays, turnover = get_rays(src, ior, grad_ior, rmax, z_range, mesh)
    
    coords = rays.swapaxes(1, 2) # Sort into coordinate pairs
    coords = coords[~np.isnan(coords).any(axis = 2)] # Remove NaN values, combine rays into one set of coordinates
    coords = coords[coords[:, 0].argsort()] # Sort by rvals
    
    caustic = np.full((rmax, 2), np.nan)
    
    for i in range(rmax): # Select top edge of ray family (with tolerance for different step sizes)
        mask = np.logical_and(coords[:, 0] < i + 0.5, coords[:, 0] > i - 0.5) 
        idx = np.argmax(coords[mask][:, 1]) # For r-window (i - 0.5, i + 0.5), find the largest z-value
        caustic[i] = coords[mask][idx]

    mask = caustic[:, 0] >= turnover[0, 0] # Values left of intersection with surface are unphysical; reject
    caustic = caustic[mask]
    turnover = np.append(turnover, [[0, src[1]]], axis = 0)
    turnover = np.flip(turnover, axis = 0)

    def caustic_rule(r):
        return np.interp(r, caustic[:, 0], caustic[:, 1], left = np.nan, right = np.nan)

    def turnover_rule(z):
        return np.interp(z, turnover[:, 1], turnover[:, 0], left = np.nan, right = np.nan)

    return caustic_rule, turnover_rule
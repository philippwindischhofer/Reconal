import numpy as np
from scipy import signal
import defs, utils, os, pickle, math
from propagation import TravelTimeCalculator
from scipy import integrate

def load_ttcs(mappath, channels_to_include):
    # load travel time maps
    with open(mappath, 'rb') as mapfile:
        map_data = pickle.load(mapfile)

    ttcs = {}
    for channel in channels_to_include:        
        if not channel in map_data:
            raise RuntimeError(f"Error: No travel time map available for channel {channel}!")
        ttcs[channel] = TravelTimeCalculator.FromDict(map_data[channel])
        
    return ttcs

def resample(tvals, sig, target_dt):

    source_dt = tvals[3] - tvals[2]
    assert target_dt < source_dt
    
    target_tvals = np.linspace(tvals[0], tvals[-1], int((tvals[-1] - tvals[0]) / target_dt))
    os_factor = int(source_dt / target_dt) + 4

    # oversample the original waveform
    os_length = os_factor * len(sig)
    os_sig = signal.resample(sig, os_length)
    os_tvals = np.linspace(tvals[0], tvals[-1] + tvals[1] - tvals[0], os_length, endpoint = False)

    # evaluate the oversampled waveform on the target grid
    target_sig = np.interp(target_tvals, os_tvals, os_sig)
    return target_tvals, target_sig

def to_antenna_rz_coordinates(pos, antenna_pos):
    local_r = np.linalg.norm(pos[:, :2] - antenna_pos[:2], axis = 1)
    local_z = pos[:, 2]    
    return np.stack([local_r, local_z], axis = -1)
    
def get_maxcorr_point(intmap):

    mapdata = intmap["map"]    
    maxind = np.unravel_index(np.argmax(mapdata), mapdata.shape)

    #change to "z", "r" for RZ maps 
    #change to "elevation", "azimuth" for angular maps
    
    maxcorr_point = {"x": intmap["x"][maxind[0]],
                     "y": intmap["y"][maxind[1]],
                     "z": intmap["z"][maxind[2]]}

    return maxcorr_point

def ang_to_cart(elevation, azimuth, radius, origin_xyz):

    xx = radius * np.cos(elevation) * np.cos(azimuth)
    yy = radius * np.cos(elevation) * np.sin(azimuth)
    zz = radius * np.sin(elevation)

    xyz = np.stack([xx, yy, zz], axis = -1) + origin_xyz
    return xyz

def cart_to_ang(xyz, origin_xyz):    
    
    xyz_rel = xyz - origin_xyz
    
    r_xy = np.linalg.norm(xyz_rel[:,:2], axis = 1)
    azimuth = np.arctan2(xyz_rel[:,1], xyz_rel[:,0])
    elevation = np.arctan2(xyz_rel[:,2], r_xy)
    
    return elevation, azimuth

#for r_xy and z maps with fixed azimuth

def ang2_to_cart(z, r, azimuth, origin_xyz):

    xx = r * np.cos(azimuth)
    yy = r * np.sin(azimuth)
    zz = z

    xyz = np.stack([xx, yy, zz], axis = -1) + origin_xyz
    return xyz

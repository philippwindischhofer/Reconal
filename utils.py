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

def corr_score_batched(sig_a, sig_b, tvals_a, tvals_b, t_ab, upsample = 5, batch_size = 1000):
    num_batches = math.ceil(len(t_ab) / batch_size)    
    t_ab_batches = np.array_split(t_ab, num_batches)
    scores = [corr_score(sig_a, sig_b, tvals_a, tvals_b, t_ab_batch, upsample = upsample) for t_ab_batch in t_ab_batches]
    return np.concatenate(scores, axis = 0)

def corr_score(sig_a, sig_b, tvals_a, tvals_b, t_ab, upsample = 10):

    # upsample both signals onto a fine grid
    target_dt = min(tvals_a[1] - tvals_a[0], tvals_b[1] - tvals_b[0]) / upsample
    
    sig_a_tvals_rs, sig_a_rs = resample(tvals_a, sig_a, target_dt)
    sig_b_tvals_rs, sig_b_rs = resample(tvals_b, sig_b, target_dt)
    
    eval_t = sig_a_tvals_rs - np.tile(t_ab, reps = (len(sig_a_tvals_rs), 1)).transpose()
    sig_b_rs_algnd = np.interp(eval_t, sig_b_tvals_rs, sig_b_rs, left = 0.0, right = 0.0)
    return np.mean(np.multiply(sig_a_rs, sig_b_rs_algnd), axis = 1) / (np.std(sig_a_rs) * np.std(sig_b_rs_algnd, axis = 1))

def to_antenna_rz_coordinates(pos, antenna_pos):
    local_r = np.linalg.norm(pos[:, :2] - antenna_pos[:2], axis = 1)
    local_z = pos[:, 2]    
    return np.stack([local_r, local_z], axis = -1)
    
def calc_relative_time(ch_a, ch_b, src_pos, ttcs, channel_positions, cable_delays, comp = "direct_ice"):

    # Convert to antenna-local (r, z) coordinates
    src_pos_loc_a = to_antenna_rz_coordinates(src_pos, channel_positions[ch_a])
    src_pos_loc_b = to_antenna_rz_coordinates(src_pos, channel_positions[ch_b])
    
    return ttcs[ch_a].get_travel_time(src_pos_loc_a, comp = comp) - ttcs[ch_b].get_travel_time(src_pos_loc_b, comp = comp) + \
        cable_delays[ch_a] - cable_delays[ch_b]

def get_maxcorr_path(intmap, r_range):
    # TODO: this got broken during the move to quasi-3d maps, need to fix again
    r_inds = np.argwhere(np.logical_and(intmap["r"] >= r_range[0], intmap["r"] < r_range[1])).flatten()
    r_vals = intmap["r"][r_inds]

    z_vals_fine = np.linspace(intmap["z"][0], intmap["z"][-1], 5000)

    z_corrmaxvals = []
    for r_ind in r_inds:
        mapvals = intmap["map"][r_ind,:]
        mapvals_fine = np.interp(z_vals_fine, intmap["z"], mapvals)
        z_corrmaxvals.append(z_vals_fine[np.argmax(mapvals_fine)])

    return r_vals, np.array(z_corrmaxvals)

def get_arrival_zenith(intmap, r_range):

    r_vals, z_vals = get_maxcorr_path(intmap, r_range)

    slope, _ = np.polyfit(r_vals, z_vals, 1)
    zenith = np.pi/2 - np.arctan(slope)

    return zenith

def get_maxcorr_point(intmap):

    mapdata = intmap["map"]    
    maxind = np.unravel_index(np.argmax(mapdata), mapdata.shape)
        
    maxcorr_point = {"x": intmap["x"][maxind[0]],
                     "y": intmap["y"][maxind[1]],
                     "z": intmap["z"][maxind[2]]}

    return maxcorr_point

def get_azimuth(intmap):

    maxcorr_point = utils.get_maxcorr_point(reco["intmap"])

    azimuth = np.arctan2(maxcorr_point[plot_axes[1]], maxcorr_point[plot_axes[0]])
    if azimuth < 0:
        azimuth += 2 * np.pi

    return azimuth

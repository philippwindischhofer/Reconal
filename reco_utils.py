import utils, pickle, itertools, math
import numpy as np, scipy.signal as signal
from propagation import TravelTimeCalculator

class CorrScoreProvider:
    
    def __init__(self, channel_sigvals, channel_times, channel_pairs_to_include, upsample = 10):

        self.corrs = {}
        self.tvals = {}
        
        for (ch_a, ch_b) in channel_pairs_to_include:
            tvals_a, tvals_b = channel_times[ch_a], channel_times[ch_b]
            sig_a, sig_b = channel_sigvals[ch_a], channel_sigvals[ch_b]
            
            # upsample both signals onto a common fine grid
            target_dt = min(tvals_a[1] - tvals_a[0], tvals_b[1] - tvals_b[0]) / upsample

            sig_a_tvals_rs, sig_a_rs = utils.resample(tvals_a, sig_a, target_dt)
            sig_b_tvals_rs, sig_b_rs = utils.resample(tvals_b, sig_b, target_dt)

            sig_a_rs_norm = (sig_a_rs - np.mean(sig_a_rs)) / np.std(sig_a_rs)
            sig_b_rs_norm = (sig_b_rs - np.mean(sig_b_rs)) / np.std(sig_b_rs)

            normfact = signal.correlate(np.ones(len(sig_a_rs)), np.ones(len(sig_b_rs)), mode = "full")
            corrs = signal.correlate(sig_a_rs_norm, sig_b_rs_norm, mode = "full") / normfact
            lags = signal.correlation_lags(len(sig_a_rs), len(sig_b_rs), mode = "full")
            tvals = lags * target_dt + sig_a_tvals_rs[0] - sig_b_tvals_rs[0]

            self.corrs[(ch_a, ch_b)] = corrs
            self.tvals[(ch_a, ch_b)] = tvals

    def get(self, ch_a, ch_b, t_ab):
        corrvals = self.corrs[(ch_a, ch_b)]
        tvals = self.tvals[(ch_a, ch_b)]
        return np.interp(t_ab, tvals, corrvals)
    
def calc_corr_score(channel_signals, channel_times, pts, ttcs, channel_pairs_to_include, channel_positions, cable_delays,
                    comps = ["direct_ice", "direct_air", "reflected"]):
    
    csp = CorrScoreProvider(channel_signals, channel_times, channel_pairs_to_include)

    channels = [channel for pair in channel_pairs_to_include for channel in pair]
    ind_loc = {ch: ttcs[ch].get_ind(utils.to_antenna_rz_coordinates(pts, channel_positions[ch])) for ch in channels}
    
    scores = []   
    for (ch_a, ch_b) in channel_pairs_to_include:
        sig_a = channel_signals[ch_a]
        sig_b = channel_signals[ch_b]
        tvals_a = channel_times[ch_a]
        tvals_b = channel_times[ch_b]

        for comp in comps:
            t_ab = ttcs[ch_a].get_travel_time_ind(ind_loc[ch_a], comp = comp) - ttcs[ch_b].get_travel_time_ind(ind_loc[ch_b], comp = comp) + \
                cable_delays[ch_a] - cable_delays[ch_b]            
            score = csp.get(ch_a, ch_b, t_ab)
            scores.append(np.nan_to_num(score, nan = 0.0))

    return np.mean(scores, axis = 0)

def build_interferometric_map_3d(channel_signals, channel_times, channel_pairs_to_include, channel_positions, cable_delays,
                                 coord_start, coord_end, num_pts, ttcs):

    x_vals = np.linspace(coord_start[0], coord_end[0], num_pts[0])
    y_vals = np.linspace(coord_start[1], coord_end[1], num_pts[1])
    z_vals = np.linspace(coord_start[2], coord_end[2], num_pts[2])
    
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing = 'ij')
    pts = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis = -1)

    intmap = calc_corr_score(channel_signals, channel_times, pts, ttcs, channel_pairs_to_include,
                             channel_positions = channel_positions, cable_delays = cable_delays)
    assert len(intmap) == len(pts)
    intmap = np.reshape(intmap, num_pts, order = "C")

    return x_vals, y_vals, z_vals, intmap

# all coordinates and coordinate ranges are given in natural feet
def interferometric_reco_3d(ttcs, channel_signals, channel_times, mappath,
                            coord_start, coord_end, num_pts,
                            channels_to_include, channel_positions, cable_delays):

    channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
    x_vals, y_vals, z_vals, intmap = build_interferometric_map_3d(channel_signals, channel_times, channel_pairs_to_include,
                                                                  channel_positions = channel_positions, cable_delays = cable_delays,
                                                                  coord_start = coord_start, coord_end = coord_end, num_pts = num_pts,
                                                                  ttcs = ttcs)
       
    reco_event = {
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "map": intmap
    }
    
    return reco_event

def build_interferometric_map_ang(channel_signals, channel_times, channel_pairs_to_include, channel_positions, cable_delays,
                                  rad, origin_xyz, elevation_range, azimuth_range, num_pts_elevation, num_pts_azimuth, ttcs):

    elevation_vals = np.linspace(*elevation_range, num_pts_elevation)
    azimuth_vals = np.linspace(*azimuth_range, num_pts_azimuth)

    ee, aa = np.meshgrid(elevation_vals, azimuth_vals)

    # convert to cartesian points
    pts = utils.ang_to_cart(ee.flatten(), aa.flatten(), radius = rad, origin_xyz = origin_xyz)

    intmap = calc_corr_score(channel_signals, channel_times, pts, ttcs, channel_pairs_to_include,
                             channel_positions = channel_positions, cable_delays = cable_delays)
    assert len(intmap) == len(pts)
    intmap = np.reshape(intmap, (num_pts_elevation, num_pts_azimuth), order = "C")

    return elevation_vals, azimuth_vals, intmap

def interferometric_reco_ang(ttcs, channel_signals, channel_times, mappath,
                             rad, origin_xyz, elevation_range, azimuth_range, num_pts_elevation, num_pts_azimuth,
                             channels_to_include, channel_positions, cable_delays):

    channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
    elevation_vals, azimuth_vals, intmap = build_interferometric_map_ang(channel_signals, channel_times, channel_pairs_to_include,
                                                                         channel_positions = channel_positions, cable_delays = cable_delays,
                                                                         rad = rad, origin_xyz = origin_xyz, elevation_range = elevation_range, azimuth_range = azimuth_range,
                                                                         num_pts_elevation = num_pts_elevation, num_pts_azimuth = num_pts_azimuth, ttcs = ttcs)

    reco_event = {
        "elevation": elevation_vals,
        "azimuth": azimuth_vals,
        "radius": rad,
        "map": intmap
    }
    
    return reco_event

#for r_xy and z maps with fixed azimuth 

def build_interferometric_map_ang2(channel_signals, channel_times, channel_pairs_to_include, channel_positions, cable_delays,
                                  azimuth, origin_xyz, z_range, r_range, num_pts_z, num_pts_r, ttcs):

    z_vals = np.linspace(*z_range, num_pts_z)
    r_vals = np.linspace(*r_range, num_pts_r)

    zz, rr = np.meshgrid(z_vals, r_vals)

    # convert to cartesian points
    pts = utils.ang2_to_cart(zz.flatten(), rr.flatten(), azimuth = azimuth, origin_xyz = origin_xyz)

    intmap = calc_corr_score(channel_signals, channel_times, pts, ttcs, channel_pairs_to_include,
                             channel_positions = channel_positions, cable_delays = cable_delays)
    assert len(intmap) == len(pts)
    intmap = np.reshape(intmap, (num_pts_z, num_pts_r), order = "C")

    return z_vals, r_vals, intmap

def interferometric_reco_ang2(ttcs, channel_signals, channel_times, mappath,
                             azimuth, origin_xyz, z_range, r_range, num_pts_z, num_pts_r,
                             channels_to_include, channel_positions, cable_delays):

    channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
    z_vals, r_vals, intmap = build_interferometric_map_ang2(channel_signals, channel_times, channel_pairs_to_include,
                                                                         channel_positions = channel_positions, cable_delays = cable_delays,
                                                                         azimuth = azimuth, origin_xyz = origin_xyz, z_range = z_range, r_range = r_range,
                                                                         num_pts_z = num_pts_z, num_pts_r = num_pts_r, ttcs = ttcs)

    reco_event = {
        "z": z_vals,
        "r": r_vals,
        "azimuth": azimuth,
        "map": intmap
    }
    
    return reco_event

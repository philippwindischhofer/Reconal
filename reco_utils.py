import utils, pickle, itertools
import numpy as np
from propagation import TravelTimeCalculator

def calc_corr_score(channel_signals, channel_times, pts, ttcs, channel_pairs_to_include, channel_positions, cable_delays,
                    comps = ["direct_ice", "direct_air", "reflected"]):
    
    scores = []
    for (ch_a, ch_b) in channel_pairs_to_include:
        print(f"channel comparison: {ch_a}<-->{ch_b}")
        sig_a = channel_signals[ch_a]
        sig_b = channel_signals[ch_b]
        tvals_a = channel_times[ch_a]
        tvals_b = channel_times[ch_b]
        
        for comp in comps:
            t_ab = utils.calc_relative_time(ch_a, ch_b, src_pos = pts, ttcs = ttcs, comp = comp,
                                            channel_positions = channel_positions, cable_delays = cable_delays)
            score = utils.corr_score_batched(sig_a, sig_b, tvals_a, tvals_b, t_ab)            
            scores.append(np.nan_to_num(score, nan = 0.0))
    
    return np.mean(scores, axis = 0)

def build_interferometric_map_2d(channel_signals, channel_times, channel_pairs_to_include, channel_positions, cable_delays,
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
def interferometric_reco(channel_signals, channel_times, outpath, mappath,
                         coord_start, coord_end, num_pts,
                         channels_to_include, channel_positions, cable_delays):

    ttcs = utils.load_ttcs(mappath, channels_to_include)
        
    # build reconstruction map in the xz-plane
    channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
    x_vals, y_vals, z_vals, intmap = build_interferometric_map_2d(channel_signals, channel_times, channel_pairs_to_include,
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

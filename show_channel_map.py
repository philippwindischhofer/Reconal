import argparse, pickle, defs, os, utils
import numpy as np
from detector import Detector
from plotting_utils import make_direction_plot

def get_azimuth(src_pos, antenna_pos):
    rel_pos = antenna_pos[:2] - src_pos[:2]
    return np.arctan2(rel_pos[1], rel_pos[0])

def antenna_locations_to_src_frame(ttcs, src_pos_xyz, channel_positions, channels_to_include, ref_channel):

    elevations_deg = []
    azimuths_deg = []
    
    ref_azimuth = get_azimuth(src_pos_xyz, channel_positions[ref_channel])

    for channel in channels_to_include:

        # convert to antenna-local coordinates
        src_pos_loc = utils.to_antenna_rz_coordinates(np.array([src_pos_xyz]), channel_positions[channel])

        travel_time = ttcs[channel].get_travel_time(src_pos_loc, comp = "direct_ice")
        tangent_vec = ttcs[channel].get_tangent_vector(src_pos_loc, comp = "direct_ice")

        assert len(tangent_vec) == 1
        
        tangent_vec = tangent_vec[0]
        elevation = np.arctan2(tangent_vec[1], -tangent_vec[0])
        azimuth = np.diff(np.unwrap([ref_azimuth, get_azimuth(src_pos_xyz, channel_positions[channel])]))[0]

        elevations_deg.append(np.rad2deg(elevation))
        azimuths_deg.append(np.rad2deg(azimuth))

    return elevations_deg, azimuths_deg

def show_channel_map(detectorpath, mappath, outdir, station_id, channels_to_include):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ttcs = utils.load_ttcs(mappath, channels_to_include)
        
    det = Detector(detectorpath)
    channel_positions = det.get_channel_positions(station_id, channels_to_include)

    src_pos_xyz = np.array([-50 / defs.cvac, 0.0 / defs.cvac, -5.0 / defs.cvac])
    elevations_deg, azimuths_deg = antenna_locations_to_src_frame(ttcs, src_pos_xyz, channel_positions, channels_to_include,
                                                                  ref_channel = 30)

    channel_labels = [f"CH{channel}" for channel in channels_to_include]
    
    outpath = os.path.join(outdir, "channel_map.pdf")
    make_direction_plot(outpath, elevations_deg, azimuths_deg, labels = channel_labels)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--detector", action = "store", dest = "detectorpath")
    parser.add_argument("--maps", action = "store", dest = "mappath") 
    parser.add_argument("--channels", nargs = "+", action = "store", dest = "channels_to_include",
                        default = [0, 1, 2, 3, 5, 6, 7, 20, 9, 10, 22, 23, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45])
    parser.add_argument("--station", type = int, default = 14, dest = "station_id")
    args = vars(parser.parse_args())

    show_channel_map(**args)

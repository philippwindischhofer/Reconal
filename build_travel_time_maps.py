import argparse, pickle, defs
from propagation import TravelTimeCalculator
from detector import Detector
import numpy as np

def build_travel_time_maps(outpath, channel_positions, z_range = (-650, 150), r_max = 1100, num_pts_z = 1000, num_pts_r = 1000,
                           ior_model = defs.ior_exp1):

    z_range_map = (z_range[0] - 1, z_range[1] + 1)
    r_max_map = r_max + 1
    
    mapdata = {}
    for channel, xyz in channel_positions.items():
        
        ttc = TravelTimeCalculator(tx_z = xyz[2],
                                   z_range = z_range_map,
                                   r_max = r_max_map,
                                   num_pts_z = 5 * num_pts_z,
                                   num_pts_r = 5 * num_pts_r)
        ttc.set_ior_and_solve(ior_model)

        mapdata[channel] = ttc.to_dict()

        print(f"Built travel time map for channel {channel}")

    with open(outpath, 'wb') as outfile:
        pickle.dump(mapdata, outfile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath", action = "store", dest = "outpath")
    parser.add_argument("--detector", action = "store", dest = "detectorpath")
    parser.add_argument("--channels", nargs = "+", action = "store", dest = "channels_to_include", default = [0], type = int)
    parser.add_argument("--station", type = int, default = 11, dest = "station_id")
    args = parser.parse_args()

    det = Detector(args.detectorpath)
    channel_positions = det.get_channel_positions(args.station_id, args.channels_to_include)
    build_travel_time_maps(args.outpath, channel_positions)

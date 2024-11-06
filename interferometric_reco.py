import argparse, pickle, defs, utils, reco_utils, os
import numpy as np
from detector import Detector

def worker_rz(wargs):
    eventpath = wargs["eventpath"]
    outdir = wargs["outdir"]
    mappath = wargs["mappath"]
    res = wargs["resolution"]
    det = wargs["detector"]

    basename = os.path.splitext(os.path.basename(eventpath))[0]
    outpath = os.path.join(outdir, f"{basename}_rz.pkl")
    print(f"Reconstructing {eventpath} -> {outpath}")

    channels_to_include = [0, 1, 2, 3, 6, 7]
    channel_positions = det.get_channel_positions(station_id = 11, channels = channels_to_include)
    cable_delays = det.get_cable_delays(station_id = 11, channels = channels_to_include)

    with open(eventpath, 'rb') as eventfile:
        event = pickle.load(eventfile)
        channel_signals = event["signals"]
        channel_times = event["times"]
        
    # center reconstruction map around PA
    PA_string_pos = channel_positions[1]
    PA_string_pos[2] = 0.0
    
    # pick some reasonable domain
    z_range = (-500, 150)
    r_max = 1000
    
    coord_start = [PA_string_pos[0],         PA_string_pos[1], z_range[0]]
    coord_end =   [PA_string_pos[0] + r_max, PA_string_pos[1], z_range[1]]
    
    reco = reco_utils.interferometric_reco_3d(channel_signals, channel_times, mappath,
                                              coord_start = coord_start, coord_end = coord_end, num_pts = [res, 1, res],
                                              channels_to_include = channels_to_include, channel_positions = channel_positions, cable_delays = cable_delays)

    with open(outpath, 'wb') as outfile:
        pickle.dump(reco, outfile)

def worker_xy(wargs):
    eventpath = wargs["eventpath"]
    outdir = wargs["outdir"]
    mappath = wargs["mappath"]
    res = wargs["resolution"]
    det = wargs["detector"]

    with open(eventpath, 'rb') as eventfile:
        event = pickle.load(eventfile)
        channel_signals = event["signals"]
        channel_times = event["times"]

    basename = os.path.splitext(os.path.basename(eventpath))[0]
    
    outpath = os.path.join(outdir, f"{basename}_xy.pkl")
    print(f"Reconstructing {eventpath} -> {outpath}")
    
    x_range = (0, 150)
    y_range = (-150, 0)

    z_slice = -80 / defs.cvac
    
    coord_start = [x_range[0], y_range[0], z_slice]
    coord_end =   [x_range[1], y_range[1], z_slice]
    
    channels_to_include = [0, 1, 2, 3, 22, 23]
    channel_positions = det.get_channel_positions(station_id = 11, channels = channels_to_include)
    cable_delays = det.get_cable_delays(station_id = 11, channels = channels_to_include)
 
    reco = reco_utils.interferometric_reco_3d(channel_signals, channel_times, mappath,
                                              coord_start = coord_start, coord_end = coord_end, num_pts = [res, res, 1],
                                              channels_to_include = channels_to_include,
                                              channel_positions = channel_positions, cable_delays = cable_delays)
    
    with open(outpath, 'wb') as outfile:
        pickle.dump(reco, outfile)

def worker_ang(wargs):
    eventpath = wargs["eventpath"]
    outdir = wargs["outdir"]
    mappath = wargs["mappath"]
    res = wargs["resolution"]
    det = wargs["detector"]

    with open(eventpath, 'rb') as eventfile:
        event = pickle.load(eventfile)
        channel_signals = event["signals"]
        channel_times = event["times"]

    basename = os.path.splitext(os.path.basename(eventpath))[0]

    outpath = os.path.join(outdir, f"{basename}_ang.pkl")
    print(f"Reconstructing {eventpath} -> {outpath}")

    channels_to_include = [0, 1, 2, 3, 22, 23]
    channel_positions = det.get_channel_positions(station_id = 11, channels = channels_to_include)
    cable_delays = det.get_cable_delays(station_id = 11, channels = channels_to_include)

    azimuth_range = (-np.pi, np.pi)
    elevation_range = (-np.pi/2, np.pi/2)
    radius = 38 / defs.cvac
    origin_xyz = channel_positions[0]  # use PA CH0- as origin of the coordinate system

    reco = reco_utils.interferometric_reco_ang(channel_signals, channel_times, mappath,
                                               rad = radius, origin_xyz = origin_xyz, elevation_range = elevation_range, azimuth_range = azimuth_range,
                                               num_pts_elevation = res, num_pts_azimuth = res, channels_to_include = channels_to_include,
                                               channel_positions = channel_positions, cable_delays = cable_delays)
    
    with open(outpath, 'wb') as outfile:
        pickle.dump(reco, outfile)
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--events", action = "store", dest = "eventpaths", nargs = "+")
    parser.add_argument("--detector", action = "store", dest = "detectorpath")
    parser.add_argument("--maps", action = "store", dest = "mappath")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--workers", action = "store", type = int, dest = "num_workers", default = 4)
    parser.add_argument("--resolution", action = "store", type=int, dest = "resolution", default = 500)
    parser.add_argument("--rz", action = "store_true", dest = "do_rz", default = False)
    parser.add_argument("--xy", action = "store_true", dest = "do_xy", default = False)
    parser.add_argument("--ang", action = "store_true", dest = "do_ang", default = False)
    args = parser.parse_args()

    outdir = args.outdir
    eventpaths = args.eventpaths
    mappath = args.mappath
    num_workers = args.num_workers
    det = Detector(args.detectorpath)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    from multiprocessing import Pool
    
    wargs = [{"eventpath": eventpath, "outdir": outdir, "mappath": mappath, "resolution": args.resolution, "detector": det} for eventpath in eventpaths]

    if args.do_rz:
        with Pool(num_workers) as p:
            p.map(worker_rz, wargs)

    if args.do_xy:
        with Pool(num_workers) as p:
            p.map(worker_xy, wargs)

    if args.do_ang:
        with Pool(num_workers) as p:
            p.map(worker_ang, wargs)

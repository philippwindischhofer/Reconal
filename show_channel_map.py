import argparse, pickle, defs, os, utils, functools
import numpy as np
from detector import Detector
from plotting_utils import make_direction_plot, make_xy_plot

def get_azimuth(src_pos, antenna_pos):
    rel_pos = antenna_pos[:2] - src_pos[:2]
    return np.arctan2(rel_pos[1], rel_pos[0])

def antenna_locations_to_src_frame(ttcs, src_pos_xyz, channel_positions, channels_to_include, ref_channel):

    elevations_deg = []
    azimuths_deg = []
    travel_times_ns = []
    
    ref_azimuth = get_azimuth(src_pos_xyz, channel_positions[ref_channel])

    for channel in channels_to_include:

        # convert to antenna-local coordinates
        src_pos_loc = utils.to_antenna_rz_coordinates(np.array([src_pos_xyz]), channel_positions[channel])

        travel_time = ttcs[channel].get_travel_time(src_pos_loc, comp = "direct_ice")
        travel_times_ns.append(travel_time)
        
        tangent_vec = ttcs[channel].get_tangent_vector(src_pos_loc, comp = "direct_ice")

        assert len(tangent_vec) == 1
        
        tangent_vec = tangent_vec[0]
        elevation = np.arctan2(tangent_vec[1], -tangent_vec[0])
        azimuth = np.diff(np.unwrap([ref_azimuth, get_azimuth(src_pos_xyz, channel_positions[channel])]))[0]

        elevations_deg.append(np.rad2deg(elevation))
        azimuths_deg.append(np.rad2deg(azimuth))

    return elevations_deg, azimuths_deg, travel_times_ns

def show_det_xy_proj(outpath, src_pos, channel_positions, station_id, channels_to_include):

    pos_x = []
    pos_y = []
    
    for channel in channels_to_include:
        pos = channel_positions[channel]
        pos_x.append(pos[0] * defs.cvac)
        pos_y.append(pos[1] * defs.cvac)

    colors = ["black"] * len(pos_x) + ["tab:red"]
        
    pos_x.append(src_pos[0] * defs.cvac)
    pos_y.append(src_pos[1] * defs.cvac)
    
    make_xy_plot(outpath, pos_x, pos_y, colors = colors, xlabel = "x [m]", ylabel = "y [m]")

def get_cherenkov_zenith(ior):
    return np.arccos(1.0 / ior)

def zenith_to_elevation(zenith):
    return np.pi/2 - zenith
    
def show_cone(ax, src_pos_xyz, ior_model = defs.ior_exp3, fs = 13):
    shallow_ior = ior_model(np.array([src_pos_xyz[2]]))
    cone_elevation = -np.rad2deg(zenith_to_elevation(get_cherenkov_zenith(shallow_ior)))
    ax.axhline(cone_elevation, ls = "dashed", color = "gray", zorder = 1, label = "On-cone (vertical shower)")

    az_vals = np.linspace(-np.rad2deg(np.pi), np.rad2deg(np.pi), 1000)
    upper_vals = np.full_like(az_vals, cone_elevation + 20)
    lower_vals = np.full_like(az_vals, cone_elevation - 20)
    ax.fill_between(az_vals, upper_vals, lower_vals, ls = "dashed", color = "gray", alpha = 0.4, zorder = 0,
                    label = r"Near-vertical showers ($\pm 20\degree$)")

    ax.legend(frameon = False, fontsize = fs)
    
def show_channel_map(detectorpath, mappath, outdir, station_id, channels_to_include):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    det = Detector(detectorpath)
    channel_positions = det.get_channel_positions(station_id, channels_to_include)

    src_pos_xyz = np.array([-50 / defs.cvac, 0.0 / defs.cvac, -5.0 / defs.cvac])
    
    outpath = os.path.join(outdir, "detector.pdf")
    show_det_xy_proj(outpath, src_pos_xyz, channel_positions, station_id, channels_to_include)
        
    ttcs = utils.load_ttcs(mappath, channels_to_include)

    elevations_deg, azimuths_deg, travel_times_ns = antenna_locations_to_src_frame(ttcs, src_pos_xyz, channel_positions, channels_to_include,
                                                                                   ref_channel = 30)
    
    channel_labels = [f"CH{channel}" for channel in channels_to_include]
    
    outpath = os.path.join(outdir, "channel_map.pdf")
    make_direction_plot(outpath, elevations_deg, azimuths_deg, travel_times_ns, obs_label = "Propagation time [ns]",
                        labels = channel_labels, epilog = functools.partial(show_cone, src_pos_xyz = src_pos_xyz))
    
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

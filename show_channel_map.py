import argparse, pickle, defs, os, utils, functools, itertools
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

def show_det_xy_proj(outpath, src_pos, channel_positions, station_id):

    def _package_channels(channels):           
        pos_x = []
        pos_y = []    
        for channel in channels:
            pos = channel_positions[channel]
            pos_x.append(pos[0] * defs.cvac)
            pos_y.append(pos[1] * defs.cvac)            
        return (pos_x, pos_y)

    labels = ["Station 14", "Dense string", "Source"]
    data_series = [_package_channels([0, 9, 22]),
                   _package_channels([30]),
                   ([src_pos[0] * defs.cvac], [src_pos[1] * defs.cvac])]
    colors = ["black", "tab:blue", "tab:red"]        
    make_xy_plot(outpath, data_series, labels, colors, xlabel = "x [m]", ylabel = "y [m]")

def get_cherenkov_zenith(ior):
    return np.arccos(1.0 / ior)

def zenith_to_elevation(zenith):
    return np.pi/2 - zenith

def get_rotation_matrix(src_vec, dest_vec):
    src_vec = src_vec / np.linalg.norm(src_vec)
    dest_vec = dest_vec / np.linalg.norm(dest_vec)
    
    v = np.cross(src_vec, dest_vec)
    c = np.dot(src_vec, dest_vec)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix

def get_cone_azimuth_elevation(shower_axis, half_opening_angle, num_pts = 100):

    def _get_azimuth_vec(vec):
        return get_azimuth(np.zeros_like(vec), vec)

    def _get_elevation_vec(vec):
        return np.arctan2(vec[2], np.sqrt(np.square(vec[0]) + np.square(vec[1])))
    
    vert_shower_axis = np.array([0, 0, 1])
    rotmat = get_rotation_matrix(vert_shower_axis, shower_axis)
    
    # generate cone vectors for vertical shower axis ...
    az_vals = np.linspace(-np.pi, np.pi, num_pts)
    on_cone_vert = [np.array([np.cos(az) * np.sin(half_opening_angle),
                              np.sin(az) * np.sin(half_opening_angle),
                              np.cos(half_opening_angle)]) for az in az_vals]

    # ... and rotate them to the real shower axis
    on_cone = [np.dot(rotmat, vec) for vec in on_cone_vert]

    # extract azimuth and elevation for each vector
    azimuth_vals = np.array([_get_azimuth_vec(vec) for vec in on_cone])
    elevation_vals = np.array([_get_elevation_vec(vec) for vec in on_cone])

    sorter = np.argsort(azimuth_vals)    
    return azimuth_vals[sorter].flatten(), elevation_vals[sorter].flatten()

def show_cone(ax, src_pos_xyz, ior_model = defs.ior_exp3, fs = 13, num_shower = 100):
    shallow_ior = ior_model(np.array([src_pos_xyz[2]]))[0]
    cone_half_opening_angle = get_cherenkov_zenith(shallow_ior)

    shower_azs = np.random.uniform(-np.pi, np.pi, num_shower)
    shower_zens = np.random.uniform(0.0, np.deg2rad(25.0), num_shower)

    ax.axhline(np.rad2deg(get_cherenkov_zenith(shallow_ior) - np.pi/2), color = "gray", zorder = 0, lw = 2, ls = "dashed",
               label = "On cone (vertical shower)")
    
    for ind, (shower_az, shower_zen) in enumerate(zip(shower_azs, shower_zens)):
        shower_axis = np.array([np.sin(shower_zen) * np.cos(shower_az),
                                np.sin(shower_zen) * np.sin(shower_az),
                                np.cos(shower_zen)])
        az_vals, elev_vals = get_cone_azimuth_elevation(shower_axis, cone_half_opening_angle)
        az_vals = np.rad2deg(az_vals)
        elev_vals = -np.rad2deg(elev_vals)

        if ind == 0:    
            ax.plot(az_vals, elev_vals, lw = 0.5, color = "gray", zorder = 0, label = r"On-cone ($\pm 25\degree$ off-vertical showers)")
        else:
            ax.plot(az_vals, elev_vals, lw = 0.5, color = "gray", zorder = 0)
    
    ax.legend(frameon = False, fontsize = fs)
    
def show_channel_map(detectorpath, mappath, outdir, station_id, channels_to_include):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    det = Detector(detectorpath)
    channel_positions = det.get_channel_positions(station_id, channels_to_include)

    src_pos_xyz = np.array([-10 / defs.cvac, 15.0 / defs.cvac, -5.0 / defs.cvac])
    
    outpath = os.path.join(outdir, "detector.pdf")
    show_det_xy_proj(outpath, src_pos_xyz, channel_positions, station_id)
        
    ttcs = utils.load_ttcs(mappath, channels_to_include)

    elevations_deg, azimuths_deg, travel_times_ns = antenna_locations_to_src_frame(ttcs, src_pos_xyz, channel_positions, channels_to_include,
                                                                                   ref_channel = 30)
    
    channel_labels = [f"{-channel_positions[channel][2]*defs.cvac:.0f}m" for channel in channels_to_include]
    
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

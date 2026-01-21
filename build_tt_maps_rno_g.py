import argparse, os, warnings, defs
from propagation import TravelTimeCalculator
from detector import Detector

def build_tt_maps_rno_g(outdir, channel_positions, z_min, z_max, r_max, num_pts_z, num_pts_r, ior_model, grad_ior_model, icestr, station_id, early_only, z_bounds):

    for channel, xyz in channel_positions.items():
        ttc = TravelTimeCalculator(tx_z = xyz[2],
                                   z_min = z_min,
                                   z_max = z_max,
                                   r_max = r_max,
                                   num_pts_z = num_pts_z,
                                   num_pts_r = num_pts_r)
        ttc.set_ior_and_solve(ior_model, grad_ior_model, 20, z_bounds, early_only = early_only)

        outpath = os.path.join(outdir, f'st{station_id}_ch{channel}_table')
        ttc.to_npz(outpath, icestr)
        print(f"Built travel time maps for channel {channel}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--detector", action = "store", dest = "detectorpath")
    parser.add_argument("--channels", type = int, nargs = "+", action = "store", dest = "channels_to_include", default = [0, 1, 2, 3, 5, 6, 7, 22, 23])
    parser.add_argument("--station", type = int, default = 11, dest = "station_id")

    # Geometry setup
    parser.add_argument("--z_min", type = float, action = "store", dest = "z_min", default = -999)
    parser.add_argument("--z_max", type = float, action = "store", dest = "z_max", default = 3)
    parser.add_argument("--r_max", type = float, action = "store", dest = "r_max", default = 1000)
    parser.add_argument("--dz", type = float, action = "store", dest = "dz", default = 0.5)
    parser.add_argument("--dr", type = float, action = "store", dest = "dr", default = 0.5)

    parser.add_argument("--early_only", action = "store_true")  # Do not generate reflected/refracted maps
    parser.add_argument("--greenland_simple", action = "store_true", dest = "simple", default = False)

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    if args.simple:
        try:
            from NuRadioMC.utilities import medium
        except ImportError:
            raise ImportError('Missing NuRadioMC package')
        ior, grad_ior = defs.get_ior_from_nuradio(medium.greenland_simple())
        z_bounds = [0.0]
        icestr = medium.greenland_simple.__name__ # Name of ice model for storage
    else:
        ior, grad_ior = defs.ior_exp3, defs.grad_ior_exp3
        z_bounds = [0.0, -14.9, -80.5]
        icestr = 'greenland_exp_3'

    if (args.dz > 0.5 or args.dr > 0.5) and not args.early_only:
        warnings.warn(f'Step sizes greater than 0.5 m may lead to refracted inaccuracies. Your step sizes are dr = {args.dr}, dz = {args.dz}.', category=RuntimeWarning)

    npts_z = int((args.z_max - args.z_min) / args.dz + 1)
    npts_r = int(args.r_max / args.dr + 1)

    det = Detector(args.detectorpath)
    channel_positions = det.get_channel_positions(args.station_id, args.channels_to_include)
    build_tt_maps_rno_g(args.outdir, channel_positions, args.z_min, args.z_max, args.r_max, npts_z, npts_r,
                        ior, grad_ior, icestr, args.station_id, args.early_only, z_bounds)
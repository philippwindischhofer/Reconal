import argparse, utils, defs, itertools
import numpy as np
import NuRadioMC.SignalProp.analyticraytracing as analytic_ray_tracing
import NuRadioMC.SignalProp.propagation as nrmc_propagation
from NuRadioMC.utilities import medium, medium_base

def get_travel_time_straightline(dest_pos, src_pos, ior_func):
    zvals = np.linspace(dest_pos[-1], src_pos[-1], 1000)
    avg_ior = np.mean(ior_func(zvals))
    dist = np.linalg.norm(dest_pos - src_pos)
    return dist * avg_ior

def test_propagation(raytracer, mappath,
                     src_pos = [100.0 / defs.cvac, 0.0 / defs.cvac, -5.0 / defs.cvac]):

    dest_channel = 0
    dest_channel_pos = np.array([0, 0, -100.0 / defs.cvac])
    
    ttcs = utils.load_ttcs(mappath, [dest_channel])

    if not isinstance(dest_channel_pos, np.ndarray):
        dest_channel_pos = np.array(dest_channel_pos)

    if not isinstance(src_pos, np.ndarray):
        src_pos = np.array(src_pos)

    # make sure everything is in SI units
    dest_channel_pos_SI = dest_channel_pos * defs.cvac
    src_pos_SI = src_pos * defs.cvac

    nrmc_to_ttc_ray_type = {
        "direct": "direct_ice",
        "reflected": "reflected",
        "refracted": "XYZ"
    }
    src_pos_loc = utils.to_antenna_rz_coordinates(np.array([src_pos]), dest_channel_pos)
    
    # perform NuRadioMC raytracing
    raytracer.set_start_and_end_point(src_pos_SI, dest_channel_pos_SI)
    raytracer.find_solutions()
    num_solutions = raytracer.get_number_of_solutions()
    for iS in range(num_solutions):
        solution_type = nrmc_propagation.solution_types[raytracer.get_solution_type(iS)]

        if solution_type == "refracted":
            continue

        # calculate the ray tangent vectors at the source location with NuRadioMC and the travel time maps
        launch_vector_nrmc_xyz = raytracer.get_launch_vector(iS)
        launch_vector_nrmc_rz = np.array([-np.linalg.norm(launch_vector_nrmc_xyz[:-1]), launch_vector_nrmc_xyz[-1]])
        launch_vector_ttcs_rz = ttcs[dest_channel].get_tangent_vector(src_pos_loc, comp = nrmc_to_ttc_ray_type[solution_type])

        # ensure unit normaliztion
        launch_vector_nrmc_rz /= np.linalg.norm(launch_vector_nrmc_rz)
        launch_vector_ttcs_rz /= np.linalg.norm(launch_vector_ttcs_rz)

        deviation = np.linalg.norm(launch_vector_nrmc_rz - launch_vector_ttcs_rz)
        print(f"{src_pos_SI} -> {dest_channel_pos_SI} ({solution_type}): deviation(launch vector) = {deviation}")
        if deviation > 1e-3:
            raise RuntimeError(f"Large deviation found for {solution_type} -> NuRadioMC: {launch_vector_nrmc_rz} vs. ttcs: {launch_vector_ttcs_rz}")

        travel_time_nrmc = raytracer.get_travel_time(iS)
        travel_time_ttcs = ttcs[dest_channel].get_travel_time(src_pos_loc, comp = nrmc_to_ttc_ray_type[solution_type])[0]
        print(f"{src_pos_SI} -> {dest_channel_pos_SI} ({solution_type}): nrmc = {travel_time_nrmc:.2f}ns, ttcs = {travel_time_ttcs:.2f}ns, diff = {travel_time_ttcs - travel_time_nrmc:.2f}ns")
        
    print("done")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", action = "store", dest = "mappath")
    args = parser.parse_args()

    # set up NuRadioMC raytracer
    ice = medium.ARA_2022()
    rt_config = {
        'propagation': {
            "attenuate_ice": False,
            "focusing_limit": 2,
            "focusing": True,
            "birefringence": False
        }
    }    
    raytracer = analytic_ray_tracing.ray_tracing(ice, config = rt_config)
    
    lat_values = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    depth_values = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    for x_src, y_src, depth_src in itertools.product(lat_values, lat_values, depth_values):
        test_propagation(raytracer, args.mappath,
                         src_pos = [x_src / defs.cvac, y_src / defs.cvac, -depth_src / defs.cvac])

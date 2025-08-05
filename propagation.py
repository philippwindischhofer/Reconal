import pykonal, copy
import numpy as np
from scipy.interpolate import interpn
from . import ray_utils
from time import perf_counter

class TravelTimeCalculator:

    # All coordinates here are 2d (r, z) in natural feet

    @classmethod
    def FromDict(cls, indict):
        obj = cls(**indict)
        return obj
    
    def __init__(self, tx_z, z_range, r_max, num_pts_z, num_pts_r, travel_time_maps = {}):

        self.tx_z = tx_z
        self.tx_pos = [0.0, self.tx_z]
        
        self.num_pts_z = num_pts_z
        self.num_pts_r = num_pts_r
        
        self.z_range = z_range
        self.r_max = r_max

        self.domain_start = np.array([0.0, self.z_range[0]])
        self.domain_end = np.array([self.r_max, self.z_range[1]])
        self.domain_shape = np.array([self.num_pts_r, self.num_pts_z])    

        # determine voxel size
        self.delta_r = self.r_max / self.num_pts_r
        self.delta_z = (self.z_range[1] - self.z_range[0]) / self.num_pts_z
        
        self.travel_time_maps = travel_time_maps
        self.tangent_vectors = {}
        self.comp_times = {}
        self._build_tangent_vectors()

    def to_dict(self):        
        return copy.deepcopy({
            "tx_z": self.tx_z,
            "z_range": self.z_range,
            "r_max": self.r_max,
            "num_pts_z": self.num_pts_z,
            "num_pts_r": self.num_pts_r,
            "travel_time_maps": self.travel_time_maps
        })

    def _build_tangent_vectors(self):
        for comp_name, comp_map in self.travel_time_maps.items():            
            grad_r, grad_z = np.gradient(comp_map[:,:,0], self.delta_r, self.delta_z)
            grad_vec = np.stack([grad_r, grad_z], axis = -1)
            self.tangent_vectors[comp_name] = -grad_vec # keep the negative to make it point towards the antenna
    
    def set_ior_and_solve(self, ior, grad_ior, num_big_rays, reflection_at_z = 0.0):

        def _get_solver(point = False):   # Solver over full domain
            
            if point == True:
                solver = pykonal.solver.PointSourceSolver(coord_sys = "cartesian")
            else:
                solver = pykonal.EikonalSolver(coord_sys = "cartesian")
            
            zvals = np.linspace(self.z_range[0], self.z_range[1], self.num_pts_z)
            iorslice = ior(zvals)
            iordata = np.expand_dims(np.tile(iorslice, reps = (self.num_pts_r, 1)), axis = -1)
            
            solver.velocity.min_coords = self.domain_start[0], self.domain_start[1], 0
            solver.velocity.npts = self.num_pts_r, self.num_pts_z, 1
            solver.velocity.node_intervals = self.delta_r, self.delta_z, 1       
            solver.velocity.values = 1.0 / iordata # c = 1 when distance measured in natural feet
            return solver
        
        def _get_updated_solver(old_solver, npts_ratio, r0, rf, z0 = None, zf = None, old_pixels = None):   # Solver over updated domain based on the last solver used
            solver = pykonal.EikonalSolver(coord_sys = "cartesian")

            if z0 is None or zf is None:
                z0 = min(old_pixels[0, r0, 1], old_pixels[0, rf, 1])
                zf = max(old_pixels[1, r0, 1], old_pixels[1, rf, 1])

            if r0 == rf: # Correct for case where thindex contains one point
                rf += 1

            r0_val = old_solver.traveltime.nodes[r0, 0, 0, 0]
            z0_val = old_solver.traveltime.nodes[0, z0, 0, 1]

            solver.velocity.min_coords = r0_val, z0_val, 0
            solver.velocity.npts = npts_ratio * (rf - r0), npts_ratio * (zf - z0), 1    # npts_ratio = new npts / old npts
            solver.velocity.node_intervals = old_solver.velocity.node_intervals[0] / npts_ratio, old_solver.velocity.node_intervals[1] / npts_ratio, 1

            zvals = solver.traveltime.nodes[0, :, 0, 1]
            iorslice = ior(zvals)
            iordata = np.expand_dims(np.tile(iorslice, reps = (solver.traveltime.npts[0], 1)), axis = -1)
            veldata = 1.0 / iordata

            solver.velocity.values = veldata
            
            return solver

        def _get_big_ray(tracer, caustic, ray_number, solver):
            
            if solver is not None:  # Generate big ray over solver domain
                rvals = solver.traveltime.nodes[:, 0, 0, 0]
            
            else:   # Generate big ray over full domain
                rvals = np.linspace(0, self.r_max, self.num_pts_r)

            bounds = np.interp(rvals, tracer[0][ray_number, 0], tracer[0][ray_number, 1]), np.interp(rvals, tracer[1][ray_number + 1, 0], tracer[1][ray_number + 1, 1])
            caustic_vals = caustic(rvals)
            big_ray = np.array((np.min(bounds, axis = 0), np.max(bounds, axis = 0)))
            
            if np.isfinite(caustic_vals).any(): # Replace ray bounds with caustic between points of intersection
                ind1 = np.nanargmin((caustic_vals - bounds[0]) ** 2)
                ind2 = np.nanargmin((caustic_vals - bounds[1]) ** 2) + 1
                big_ray[1][ind1 : ind2] = caustic_vals[ind1 : ind2]

            big_ray = np.swapaxes(np.stack((np.tile(rvals, (2,1)), big_ray), 1), 1, 2)

            pixels = np.array([self._coord_to_pixel(solver, big_ray[0]), self._coord_to_pixel(solver, big_ray[1])])

            return pixels
        
        def _set_soner_condition(solver, pixels):
            max_node = solver.traveltime.nodes.shape[:-1]
            soner_pixels = np.copy(pixels).reshape(-1, pixels.shape[-1])    # Compress since we don't care about max vs min pixels
            for pt in soner_pixels:
                if np.any(pt < 0) or np.any(pt >= max_node):
                    continue
                solver.traveltime.values[*pt] = np.inf
                solver.known[*pt] = True

        def _select_relevant_traveltimes(solver, pixels): # Traveltimes are unphysical outside ray boundaries
            for r in range(solver.traveltime.npts[0]):
                min_z = pixels[0, r, 1]
                max_z = pixels[1, r, 1]
                if min_z < 0:
                    min_z = 0 
                if max_z < 0:
                    max_z = 0
                solver.traveltime.values[r, :min_z + 1, 0] = np.nan
                solver.traveltime.values[r, max_z:, 0] = np.nan
        
        # Set up ray geometry
        
        boundary_z_ind = self._coord_to_pykonal([[0, reflection_at_z]])[0][1]
        caustic = ray_utils.get_caustic(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, reflection_at_z)[0]

        # Calculate rays transmitted into the air
        start = perf_counter()
        solver = _get_solver(point = True)
        solver.src_loc = 0, self.tx_z, 0
        solver.solve()
        self.travel_time_maps["direct_air"] = np.copy(solver.traveltime.values)
        self.travel_time_maps["direct_air"][:, :boundary_z_ind, :] = np.nan # this is now unphysical in the ice, as in part of the volume
                                                                            # head-waves will overtake the direct bending modes
        end = perf_counter()
        self.comp_times['direct_air'] = end - start

        # Calculate direct rays in the ice
        start = perf_counter()
        
        solver = _get_solver(point = True)
        solver.src_loc = 0, self.tx_z, 0
        
        rvals = np.arange(0, self.r_max + 1, self.delta_r)
        caustic_points = np.swapaxes(np.array((rvals, caustic(rvals))), 0, 1)[~np.isnan(caustic(rvals))]
        caustic_pixels = self._coord_to_pixel(solver, caustic_points)

        solver.known[:, boundary_z_ind:, :] = True
        _set_soner_condition(solver, caustic_pixels)

        solver.solve()
        end = perf_counter()
        self.comp_times['direct_ice'] = end - start

        self.travel_time_maps["direct_ice"] = np.copy(solver.traveltime.values)
        self.travel_time_maps["direct_ice"][:, boundary_z_ind+1:, :] = np.nan # this is now unphysical in the air
        
        # Calculate reflected rays: place a line source at the air/ice boundary
        start = perf_counter()
        solver = _get_solver()
        solver.traveltime.values[:, boundary_z_ind - 2, :] = self.travel_time_maps["direct_ice"][:, boundary_z_ind - 2, :]
        solver.unknown[:, boundary_z_ind - 2, :] = False
        for r_ind in range(self.num_pts_r):
            solver.trial.push(r_ind, boundary_z_ind - 2, 0)
        _set_soner_condition(solver, caustic_pixels)
        solver.solve()

        self.travel_time_maps["reflected"] = np.copy(solver.traveltime.values)
        self.travel_time_maps["reflected"][:, boundary_z_ind:, :] = np.nan # this is now unphysical in the air

        end = perf_counter()
        self.comp_times['reflected'] = end - start
        
        # Calculate refracted rays: big rays method
        start = perf_counter()

        # Ray tracer: calculate individual rays, turnover points, and caustic
        theta_min, theta_max = ray_utils.get_theta_min(self.tx_pos, ior, reflection_at_z), 89
        mesh = (np.linspace(theta_min + 1, theta_max - 1, num_big_rays + 1), np.linspace(theta_min + 2, theta_max, num_big_rays + 1))
        ray_data = (ray_utils.get_rays(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, mesh[0]), ray_utils.get_rays(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, mesh[1]))
        tracer = ray_data[0][0], ray_data[1][0]
        turnover = ray_data[0][1]
        S = 6 # pixel tolerance; how small a big ray can get before adaptive gridsize kicks in

        self.travel_time_maps["refracted"] = np.full((self.num_pts_r, self.num_pts_z, 1), np.nan)
        
        for i in range(num_big_rays):

            solver = _get_solver()

            pixels = _get_big_ray(tracer, caustic, i, solver)

            r1 = self._coord_to_pixel(solver, np.array([turnover[i]]))[0, 0]    # first ray turns over

            # Setting line source
            solver.traveltime.values[r1, :, 0] = np.copy(self.travel_time_maps['direct_ice'])[r1, :, 0]
            solver.known[:r1 + 1, pixels[0, r1, 1] : pixels[1, r1, 1], 0] = True
            for z_ind in range(pixels[0, r1, 1], pixels[1, r1, 1]):
                solver.trial.push(r1, z_ind, 0)

            _set_soner_condition(solver, pixels)
            solver.solve()
            _select_relevant_traveltimes(solver, pixels)

            # Adaptive grid-sizing: divide pixels into quarters where rays get too thin
            thindex = np.array(np.nonzero(pixels[1, :, 1] - pixels[0, :, 1] <= S))
            thindex = thindex[thindex >= r1]
            solvers = [solver]  # List to keep all solvers
            big_ray_list = [pixels] # List to keep big ray at various resolutions
            
            # Zoom in on pinch point
            while thindex.size > 0:
                old_solver = solvers[-1]
                old_pixels = big_ray_list[-1]
                
                r0, rf = thindex.min(), thindex.max()    # first and last points (after turnover) where ray thickness is below our tolerance
                
                z0 = min(old_pixels[0, r0, 1], old_pixels[0, rf, 1])
                zf = max(old_pixels[1, r0, 1], old_pixels[1, rf, 1])
                
                if r0 == rf: # Correct for case where thindex contains one point
                    rf += 1

                r0_val = old_solver.velocity.nodes[r0, 0, 0, 0]
                z0_val = old_solver.velocity.nodes[0, z0, 0, 1]

                solver.velocity.min_coords = r0_val, z0_val, 0
                solver.velocity.npts = 2 * (rf - r0), 2 * (zf - z0), 1
                solver.velocity.node_intervals = old_solver.velocity.node_intervals[0] / 2, old_solver.velocity.node_intervals[1] / 2, 1

                zvals = solver.traveltime.nodes[0, :, 0, 1]
                iorslice = ior(zvals)
                iordata = np.expand_dims(np.tile(iorslice, reps = (solver.traveltime.npts[0], 1)), axis = -1)
                veldata = 1.0 / iordata

                solver.velocity.values = veldata

                solver = _get_updated_solver(solvers[-1], 2, r0, rf, tracer = tracer, ray_number = i, old_pixels = pixels)

                # Get big ray
                pixels = _get_big_ray(tracer, caustic, i, solver)
                big_ray_list.append(pixels)

                # Set line source
                nodes = solver.traveltime.nodes[0, ...]
                solver.traveltime.values[0, :, 0] = old_solver.traveltime.resample(nodes.reshape(-1, 3))
                solver.known[0, :, 0] = True
                for z_ind in range(pixels[0, 0, 1], pixels[1, 0, 1]):
                    solver.trial.push(0, z_ind, 0)

                _set_soner_condition(solver, pixels)
                solver.solve()
                _select_relevant_traveltimes(solver, pixels)
                solvers.append(solver)

                thindex = np.array(np.nonzero(pixels[1, :, 1] - pixels[0, :, 1] <= S)) # Reset thindex with new grid size

                if np.any(np.isfinite(solver.traveltime.values[-1, ...])): # Check if solution has successfully propagated through pinch point
                    break

            # Return to normal grid size (zoom out)
            solvers.reverse()
            big_ray_list.reverse()

            for ii in range(1, len(solvers)): 
                
                solver = solvers[ii]
                old_solver = solvers[ii - 1]
                pixels = big_ray_list[ii]
                old_pixels = big_ray_list[ii - 1]
                
                solver.known[np.isinf(solver.traveltime.values)] = False
                solver.unknown[np.isinf(solver.traveltime.values)] = True

                r0 = self._coord_to_pixel(solver, [old_solver.vv.max_coords[:-1]])[0, 0]
                
                # Set line source
                try:
                    solver.traveltime.values[r0, pixels[0, r0, 1] : pixels[1, r0, 1], 0] = old_solver.traveltime.values[-2, old_pixels[0, -2, 1] : old_pixels[1, -2, 1] : 2, 0]
                except ValueError: # Big ray at r = r0 are different sizes across 2 solvers due to rounding
                    try: # New solver > old solver
                        solver.traveltime.values[r0, pixels[0, r0, 1] : pixels[1, r0, 1] - 1, 0] = old_solver.traveltime.values[-2, old_pixels[0, -2, 1] : old_pixels[1, -2, 1] : 2, 0]
                    except ValueError: # Old solver > new solver
                        solver.traveltime.values[r0, pixels[0, r0, 1] : pixels[1, r0, 1], 0] = old_solver.traveltime.values[-2, old_pixels[0, -2, 1] : old_pixels[1, -2, 1] - 2 : 2, 0]

                solver.unknown[r0, pixels[0, r0, 1] : pixels[1, r0, 1]] = False
                for z_ind in range(pixels[0, r0, 1], pixels[1, r0, 1]):
                    solver.trial.push(r0, z_ind, 0)

                solver.known[r0 - 1, :, 0] = True
                _set_soner_condition(solver, pixels)

                solver.solve()

                _select_relevant_traveltimes(solver, pixels)

            # Build big ray map (interpolate finer solvers back onto coarse grid)
            big_ray_map = np.full((self.num_pts_r, self.num_pts_z, 1), np.nan)
            rvals = np.linspace(self.domain_start[0], self.domain_end[0], self.domain_shape[0])
            zvals = np.linspace(self.domain_start[1], self.domain_end[1], self.domain_shape[1])
            eval_at = np.moveaxis(np.array(np.meshgrid(rvals, zvals, np.array([0]), indexing = "ij")), 0, -1)

            for solver in solvers:
                points = solver.traveltime.nodes[:, 0, 0, 0], solver.traveltime.nodes[0, :, 0, 1], solver.traveltime.nodes[0, 0, :, 2]
                values = solver.traveltime.values
                big_ray_map[~np.isfinite(big_ray_map)] = interpn(points, values, eval_at, bounds_error = False)[~np.isfinite(big_ray_map)]

            self.travel_time_maps['refracted'][~np.isfinite(self.travel_time_maps['refracted'])] = big_ray_map[~np.isfinite(self.travel_time_maps['refracted'])]
    
        self.travel_time_maps['refracted'][self.travel_time_maps['refracted'] < self.travel_time_maps['direct_ice'] + 1] = np.nan
        end = perf_counter()
        self.comp_times['refracted'] = end - start
        self._build_tangent_vectors()

    def get_ind(self, coord):
        return np.transpose(self._coord_to_pixel(coord))        
        
    def get_travel_time(self, coord, comp = "direct_ice"):

        if comp not in self.travel_time_maps:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")

        ind = self.get_ind(coord)

        return self.travel_time_maps[comp][*ind]

    def get_travel_time_ind(self, ind, comp = "direct_ice"):
        return self.travel_time_maps[comp][*ind]
    
    def get_tangent_vector(self, coord, comp = "direct_ice"):

        if comp not in self.travel_time_maps:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")

        ind_r, ind_z, _ = self.get_ind(coord)
        return self.tangent_vectors[comp][ind_r, ind_z]
    
    def _coord_to_pykonal(self, coord, solver = None):
        return tuple(self._coord_to_pixel(solver, coord))
        
    def _coord_to_pixel(self, solver, coord):
        return self._coord_to_frac_pixel(solver, coord).astype(int)
        
    def _coord_to_frac_pixel(self, solver, coord): 
        if isinstance(coord, list):
            coord = np.array(coord)

        if solver is not None:   
            start = np.array(solver.vv.min_coords[:-1])
            delta = np.array(solver.vv.node_intervals[:-1])
        
        else:
            start = self.domain_start
            delta = np.array([self.delta_r, self.delta_z])

        pixel_2d = (coord - start) / delta
        pixel_3d = np.append(pixel_2d, np.zeros((len(coord), 1)), axis = 1)
        return pixel_3d

import pykonal, copy
import numpy as np
import ray_utils
from scipy.interpolate import interpn
from scipy.constants import c

class TravelTimeCalculator:

    # All coordinates here are 2d (r, z) in meters

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
        self.delta_r = self.r_max / (self.num_pts_r - 1)
        self.delta_z = (self.z_range[1] - self.z_range[0]) / (self.num_pts_z - 1)
        
        self.travel_time_fields = {}

        if travel_time_maps:
            for comp, tt_map in travel_time_maps.items():
                self.travel_time_fields[comp] = pykonal.fields.ScalarField3D(coord_sys = 'cartesian')
                self.travel_time_fields[comp].min_coords = self.domain_start[0], self.domain_start[1], 0
                self.travel_time_fields[comp].npts = self.num_pts_r, self.num_pts_z, 1
                self.travel_time_fields[comp].node_intervals = self.delta_r, self.delta_z, 1
                self.travel_time_fields[comp].values = tt_map

    def to_dict(self):        
        return copy.deepcopy({
            "tx_z": self.tx_z,
            "z_range": self.z_range,
            "r_max": self.r_max,
            "num_pts_z": self.num_pts_z,
            "num_pts_r": self.num_pts_r,
            "travel_time_fields": self.travel_time_fields
        })
    
    def set_ior_and_solve(self, ior, grad_ior, num_big_rays = 0, reflection_at_z = 0.0):

        """
        Supports discontinuous piecewise ice models.
        
        Parameters
        __________
        ior : function
            Function which returns index of refraction n(z). Should accept floats or ndarrays of z-values (not 3d coordinates).
        grad_ior : function
            Function which returns dn/dz at depth z. See ior.
        num_big_rays : int
            Number of big rays used to calculate refracted maps. Defaults to 0 (in this case no refracted map generated)
        reflection_at_z : float, optional
            z value of surface-ice discontinuity. Defaults to z = 0.
        """
        
        speed_of_light = c / (1e9) # m/ns

        def _get_solver(point = False):    
            """
            Initializes FMM solver over the full computational domain.
            """
            
            if point:
                solver = pykonal.solver.PointSourceSolver(coord_sys = "cartesian")
            else:
                solver = pykonal.EikonalSolver(coord_sys = "cartesian")
            
            solver.velocity.min_coords = self.domain_start[0], self.domain_start[1], 0
            solver.velocity.npts = self.num_pts_r, self.num_pts_z, 1
            solver.velocity.node_intervals = self.delta_r, self.delta_z, 1

            iorslice = ior(solver.velocity.nodes[0, :, 0, 1])
            iordata = np.expand_dims(np.tile(iorslice, reps = (self.num_pts_r, 1)), axis = -1)
            solver.velocity.values = speed_of_light / iordata

            return solver

        def _get_big_ray(tracer, caustic, ray_number, solver):
            """
            Generates big ray node bounds and traveltimes according to current solver domain & mesh size.
            Default ray bounds are generated through reconal raytracer.
            """
            if solver is not None:  # Generate big ray over solver domain
                rvals = solver.traveltime.nodes[:, 0, 0, 0]
            
            else:   # Generate big ray over full domain
                rvals = np.linspace(0, self.r_max, self.num_pts_r)

            # Interpolate raytracer depths, times onto gridded r-values
            bounds = np.array([np.stack((np.interp(rvals, tracer[0][ray_number, 0], tracer[0][ray_number, 1]),  # First bounding ray (zvals)
                                         np.interp(rvals, tracer[0][ray_number, 0], tracer[0][ray_number, 2])), axis = 1),  # (traveltimes)
                               np.stack((np.interp(rvals, tracer[1][ray_number + 1, 0], tracer[1][ray_number + 1, 1]),  # Second bounding ray (zvals)
                                         np.interp(rvals, tracer[1][ray_number + 1, 0], tracer[1][ray_number + 1, 2])), axis = 1)]) # (traveltimes)
            
            # Sort into top bounding ray, bottom bounding ray
            big_ray = np.array((bounds[np.argmin(bounds[..., 0], axis = 0), np.arange(bounds.shape[1])],
                                bounds[np.argmax(bounds[..., 0], axis = 0), np.arange(bounds.shape[1])]))
            
            # Interpolate caustic depths, times onto gridded r-values
            caustic_vals = np.stack((np.interp(rvals, caustic[0], caustic[1], left = np.nan, right = np.nan),
                                     np.interp(rvals, caustic[0], caustic[2], left = np.nan, right = np.nan)), axis = 1)
            
            if np.isfinite(caustic_vals).any(): # Replace top ray bound with caustic between contact points
                ind1 = np.nanargmin(np.square(caustic_vals[:, 0] - bounds[0, :, 0]))
                ind2 = np.nanargmin(np.square(caustic_vals[:, 0] - bounds[1, :, 0])) + 1
                big_ray[1][ind1 : ind2] = caustic_vals[ind1 : ind2]

            big_ray = np.concatenate((np.expand_dims(np.tile(rvals, (2,1)), axis = 2), big_ray), axis = 2)
            big_ray_nodes = np.array([self._coord_to_node(big_ray[0, :, :-1], solver), self._coord_to_node(big_ray[1, :, :-1], solver)])
            big_ray_times = big_ray[..., -1]

            return big_ray_nodes, big_ray_times
        
        def point_solve(solver, src_ind):
            """
            Corrects for a tolerance issue & source backfilling in the pykonal PointSourceSolver script.
            """
            solver.initialize_near_field_grid()

            r0 = np.sqrt(np.sum(np.square(solver.src_loc)))
            t0 = np.pi - np.arccos(solver.src_loc[2]/ r0)
            p0 = (np.arctan2(solver.src_loc[1], solver.src_loc[0]) + np.pi) % (2 * np.pi)
            origin = (r0, t0, p0)
            nodes = solver.near_field.vv.transform_coordinates(
                solver.coord_sys,
                origin
            )
            bool_idx = True
            for iax in range(3):
                nodes[..., iax][np.abs(nodes[..., iax] - solver.vv.min_coords[iax]) < 1e-14] = solver.vv.min_coords[iax]
                bool_idx = bool_idx &(
                        (solver.vv.iax_isnull[iax])
                    |(solver.vv.iax_isperiodic[iax])
                    |(
                            (nodes[...,iax] >= solver.vv.min_coords[iax])
                        &(nodes[...,iax] <= solver.vv.max_coords[iax])
                    )
                )
            idxs = np.nonzero(bool_idx)
            solver.near_field.vv.values[idxs] = solver.vv.resample(nodes[idxs].reshape(-1, 3))

            solver.initialize_near_field_narrow_band()
            solver.near_field.solve()
            solver.interpolate_near_field_traveltime_onto_far_field()
            solver.initialize_far_field_narrow_band()
            solver.traveltime.values[0, src_ind, 0] = 0
            solver.known[np.isfinite(solver.traveltime.values)] = True
            super(pykonal.solver.PointSourceSolver, solver).solve()
        
        def _set_boundary_condition(solver, nodes, times = None):
            """
            Sets finite or infinite boundary conditions along nodes.
            """
            boundary_nodes = nodes.reshape(-1, nodes.shape[-1]) # Reshapes multiple sets of nodes (N total) into array with shape (N,3) or (N,4)
                                                                # depending on time condition (necessary for big ray generation with top/bottom bounds)
            nodes_mask = np.logical_and((boundary_nodes < solver.traveltime.nodes.shape[:-1]).all(axis = 1), (boundary_nodes >= 0).all(axis = 1))
            
            if times is not None:  # Finite traveltime condition
                times = times.reshape(-1)[nodes_mask]
                inds = tuple(boundary_nodes[nodes_mask].swapaxes(0, 1))
                solver.traveltime.values[*inds] = times
                solver.known[*inds] = True

            else:   # Infinite traveltime condition
                inds = tuple(boundary_nodes[nodes_mask].swapaxes(0, 1))
                solver.known[*inds] = True

        def _select_relevant_traveltimes(solver, nodes):
            """
            Eliminates unphysical traveltimes outside big ray boundaries.
            """
            for r in range(solver.traveltime.npts[0]):
                min_z = max(nodes[0, r, 1], 0)
                max_z = max(nodes[1, r, 1], 0)
                solver.traveltime.values[r, :min_z, 0] = np.inf
                solver.traveltime.values[r, max_z + 1:, 0] = np.inf
        
        # Set up ray geometry
        src_ind = self._coord_to_pykonal([self.tx_pos])[0][1]
        boundary_z_ind = self._coord_to_pykonal([[0, reflection_at_z]])[0][1]
        caustic, turnover_bounds, reflected_bounds = ray_utils.get_special_bounds(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, reflection_at_z)

        rvals = np.arange(0, self.r_max + 1, self.delta_r)
        caustic_bounds = np.array([rvals, np.interp(rvals, caustic[0], caustic[1], left = np.nan)]).swapaxes(0, 1)
        caustic_nodes = self._coord_to_node(caustic_bounds[np.isfinite(caustic_bounds).all(axis = 1)])

        zvals = np.arange(self.domain_start[1], reflection_at_z, self.delta_z)
        turnover_vals = np.interp(zvals, turnover_bounds[1], turnover_bounds[0], left = np.nan, right = np.nan)

        # Calculate direct rays in the ice
        solver = _get_solver(point = True)
        solver.src_loc = self.tx_pos + [0]
        solver._ntheta = 2

        _set_boundary_condition(solver, caustic_nodes)

        try:
            solver.known[:, boundary_z_ind + 2:] = True # Prevent solution from propagating into air
        except IndexError:
            pass

        point_solve(solver, src_ind)

        self.travel_time_fields['direct'] = solver.traveltime

        # Calculate rays transmitted into the air
        solver = _get_solver()
        solver.traveltime.values[:, boundary_z_ind, :] = self.travel_time_fields["direct"].values[:, boundary_z_ind, :]
        solver.unknown[:, boundary_z_ind] = False
        solver.known[:, :boundary_z_ind] = True
        for r_ind in range(self.num_pts_r):
            solver.trial.push(r_ind, boundary_z_ind, 0)
        solver.solve()

        self.travel_time_fields['direct'].values[:, boundary_z_ind + 1:] = solver.traveltime.values[:, boundary_z_ind + 1:]

        # Calculate reflected rays: place a line source at the air/ice boundary
        solver = _get_solver()
        solver.traveltime.values[:, boundary_z_ind] = self.travel_time_fields["direct"].values[:, boundary_z_ind]
        solver.unknown[:, boundary_z_ind] = False
        solver.known[:, boundary_z_ind + 1:] = True
        for r_ind in range(self.num_pts_r):
            solver.trial.push(r_ind, boundary_z_ind, 0)
        solver.solve()
        
        max_vals = np.interp(np.linspace(0, self.r_max, self.num_pts_r), reflected_bounds[0], reflected_bounds[1], left = np.nan, right = np.nan)
        max_vals[max_vals < self.domain_start[1]] = self.domain_start[1]
        r_inds = np.nonzero(np.isfinite(max_vals))[0]
        max_inds = self._coord_to_node(np.array([np.zeros_like(r_inds, dtype = np.float64), max_vals[r_inds]]).swapaxes(0, 1))[:, 1]

        for max_ind, r_ind in zip(max_inds, r_inds):
            solver.traveltime.values[r_ind, max_ind : boundary_z_ind] = np.inf # Eliminate non-raytracing solutions from reflected map
        solver.traveltime.values[r_inds[-1]:] = np.inf
        solver.traveltime.values[:, boundary_z_ind + 1:, :] = np.inf # this is now unphysical in the air
        self.travel_time_fields['reflected'] = solver.traveltime
        
        # Calculate refracted rays: big rays method

        # Ray tracer: calculate individual rays & turnover points
        theta_min, theta_max = ray_utils.get_theta_min(self.tx_pos, ior, reflection_at_z) + 0.0001, 89.9999 # Exactly 90 degrees would propagate horizontally forever
        mesh = (np.linspace(theta_min, theta_max - 5, num_big_rays + 1), np.linspace(theta_min + 5, theta_max, num_big_rays + 1))
        ray_data = (ray_utils.get_rays(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, mesh[0], step = self.delta_r),
                    ray_utils.get_rays(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, mesh[1], step = self.delta_r))
        tracer = ray_data[0][0], ray_data[1][0]

        # Set up refracted field template on full domain
        self.travel_time_fields['refracted'] = pykonal.fields.ScalarField3D(coord_sys = 'cartesian')
        self.travel_time_fields['refracted'].min_coords = self.domain_start[0], self.domain_start[1], 0
        self.travel_time_fields['refracted'].npts = self.num_pts_r, self.num_pts_z, 1
        self.travel_time_fields['refracted'].node_intervals = self.delta_r, self.delta_z, 1

        for iR in range(num_big_rays):

            solver = _get_solver()
            nodes, times = _get_big_ray(tracer, caustic, iR, solver)

            # Setting line source using turnover line
            coords = np.array([turnover_vals[src_ind:], zvals[src_ind:]]).swapaxes(0, 1)
            start_nodes = self._coord_to_node(coords)
            for px in start_nodes:
                if nodes[0, px[0], 1] < px[1] < nodes[1, px[0], 1]:
                    solver.traveltime.values[*px] = self.travel_time_fields['direct'].values[*px]
                    solver.known[:px[0] + 1, px[1]] = True
                    solver.trial.push(*px)
            
            _set_boundary_condition(solver, nodes, times)
            solver.solve()
            _select_relevant_traveltimes(solver, nodes)

            # Adaptive mesh refinement: divide nodes into quarters where big rays get too thin
            S = 5   # node tolerance; ray thickness which triggers adaptive mesh refinement
            thindex = np.array(np.nonzero(nodes[1, :, 1] - nodes[0, :, 1] <= S))
            thindex = thindex[np.logical_and(thindex >= np.max(start_nodes[:, 0]), thindex < self.num_pts_r - 1)]
            solvers = [solver]  # List to keep all solvers
            big_ray_nodes = [nodes] # List to keep big ray at various resolutions
            big_ray_times = [times] # List to keep raytracing traveltimes
            
            # Zoom in on pinch point
            while thindex.size > 0:
                old_solver = solvers[-1]
                old_nodes = big_ray_nodes[-1]
                solver = pykonal.EikonalSolver(coord_sys = 'cartesian')

                r0, rf = thindex.min(), thindex.max()    # First and last points (after turnover) where ray thickness is below our tolerance
                
                z0 = min(old_nodes[0, r0, 1], old_nodes[0, rf, 1]) - 1
                zf = max(old_nodes[1, r0, 1], old_nodes[1, rf, 1]) + 1
                
                if r0 == rf: # Correct for case where thindex contains one point
                    rf += 1

                solver.velocity.min_coords = old_solver.velocity.nodes[r0, 0, 0, 0], old_solver.velocity.nodes[0, z0, 0, 1], 0
                solver.velocity.npts = 2 * (rf - r0) + 1, 2 * (zf - z0) + 1, 1
                solver.velocity.node_intervals = old_solver.velocity.node_intervals[0] / 2, old_solver.velocity.node_intervals[1] / 2, 1

                iorslice = ior(solver.traveltime.nodes[0, :, 0, 1])
                iordata = np.expand_dims(np.tile(iorslice, reps = (solver.velocity.npts[0], 1)), axis = -1)
                solver.velocity.values = speed_of_light / iordata

                # Get big ray
                nodes, times = _get_big_ray(tracer, caustic, iR, solver)
                big_ray_nodes.append(nodes)
                big_ray_times.append(times)

                # Set line source
                source = np.squeeze(solver.traveltime.nodes[0])
                solver.traveltime.values[0, :, 0] = old_solver.traveltime.resample(source, null = np.inf)
                solver.known[0] = True
                for z_ind in range(nodes[0, 0, 1], nodes[1, 0, 1]):
                    solver.trial.push(0, z_ind, 0)

                _set_boundary_condition(solver, nodes, times)
                bounds_count = np.count_nonzero(np.isfinite(solver.traveltime.values[-1, nodes[0, -1, 1] : nodes[1, -1, 1]]))
                solver.solve()

                _select_relevant_traveltimes(solver, nodes)
                solvers.append(solver)

                thindex = np.array(np.nonzero(nodes[1, :, 1] - nodes[0, :, 1] <= S)) # Reset thindex with new mesh size

                # Check if solution has successfully propagated through pinch point
                if np.count_nonzero(np.isfinite(solver.traveltime.values[-1, nodes[0, -1, 1] : nodes[1, -1, 1]])) > bounds_count:
                    break

            # Return to normal mesh size (zoom out)
            solvers.reverse()
            big_ray_nodes.reverse()
            big_ray_times.reverse()

            for iS in range(1, len(solvers)): 
                
                solver, old_solver = solvers[iS], solvers[iS - 1]
                nodes, old_nodes = big_ray_nodes[iS], big_ray_nodes[iS - 1]
                times = big_ray_times[iS]

                r0, zf = self._coord_to_node([old_solver.vv.max_coords[:-1]], solver)[0, :-1]
                z0 = self._coord_to_node([old_solver.vv.min_coords[:-1]], solver)[0, 1]
                solver.traveltime.values[r0, z0 : zf + 1] = old_solver.traveltime.values[-1, ::2]
                solver.unknown[r0, nodes[0, r0, 1] : nodes[1, r0, 1]] = False
                solver.known[:r0] = True
                for z_ind in range(nodes[0, r0, 1], nodes[1, r0, 1]):
                    solver.trial.push(r0, z_ind, 0)

                _set_boundary_condition(solver, nodes, times)
                solver.solve()
                _select_relevant_traveltimes(solver, nodes)

            # Build big ray map (interpolate finer solvers back onto original mesh)
            big_ray_map = np.full((self.num_pts_r, self.num_pts_z, 1), np.inf)
            r = np.linspace(self.domain_start[0], self.domain_end[0], self.domain_shape[0])
            z = np.linspace(self.domain_start[1], self.domain_end[1], self.domain_shape[1])
            eval_at = np.moveaxis(np.array(np.meshgrid(r, z, np.array([0]), indexing = "ij")), 0, -1)

            for solver in solvers:
                points = solver.traveltime.nodes[:, 0, 0, 0], solver.traveltime.nodes[0, :, 0, 1], solver.traveltime.nodes[0, 0, :, 2]
                if solver == solvers[-1]:    # Initial solver is always on full domain; no interpolation necessary
                    big_ray_map[~np.isfinite(big_ray_map)] = solver.traveltime.values[~np.isfinite(big_ray_map)]
                else:
                    values = np.copy(solver.traveltime.values)
                    big_ray_map[~np.isfinite(big_ray_map)] = interpn(points, values, eval_at, bounds_error = False, fill_value = np.inf)[~np.isfinite(big_ray_map)]
            self.travel_time_fields['refracted'].values[~np.isfinite(self.travel_time_fields['refracted'].values)] = big_ray_map[~np.isfinite(self.travel_time_fields['refracted'].values)]
        
        # Eliminate duplicate direct/refracted solutions
        z_inds = np.nonzero(np.isfinite(turnover_vals))[0]
        turnover_inds = self._coord_to_node(np.array([turnover_vals[z_inds], np.zeros_like(z_inds, dtype = np.float64)]).swapaxes(0, 1))[:, 0]
        for turnover_ind, z_ind in zip(turnover_inds, z_inds):
            self.travel_time_fields["direct"].values[turnover_ind:, z_ind] = np.inf
            if z_ind > src_ind:
                self.travel_time_fields["refracted"].values[:turnover_ind, z_ind] = np.inf

    def get_ind(self, coord):
        return np.transpose(self._coord_to_node(coord))        
        
    def get_travel_time(self, coord, comp = "direct"):

        if comp not in self.travel_time_fields:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")

        ind = self.get_ind(coord)

        return self.travel_time_fields[comp].values[*ind]

    def get_travel_time_ind(self, ind, comp = "direct"):
        return self.travel_time_fields[comp].values[*ind]
    
    def get_tangent_vector(self, coord, comp = "direct"):

        if comp not in self.travel_time_fields:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")

        ind_r, ind_z, _ = self.get_ind(coord)
        return self.travel_time_fields[comp].gradient[ind_r, ind_z, 0]
    
    def _coord_to_pykonal(self, coord, solver = None):
        return tuple(self._coord_to_node(coord, solver))
        
    def _coord_to_node(self, coord, solver = None):
        return self._coord_to_frac_node(coord, solver).astype(int)
        
    def _coord_to_frac_node(self, coord, solver): 
        if isinstance(coord, list):
            coord = np.array(coord)

        if solver:   
            start = np.array(solver.vv.min_coords[:-1])
            delta = np.array(solver.vv.node_intervals[:-1])
        
        else:
            start = self.domain_start
            delta = np.array([self.delta_r, self.delta_z])

        node_2d = (coord - start) / delta
        node_3d = np.append(node_2d, np.zeros((len(coord), 1)), axis = 1)
        return node_3d

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
    
    def __init__(self, tx_z, z_min, z_max, r_max, num_pts_z, num_pts_r, travel_time_maps = {}):

        self.tx_z = tx_z
        self.tx_pos = [0.0, self.tx_z]
        
        self.num_pts_z = int(num_pts_z)
        self.num_pts_r = int(num_pts_r)
        
        self.z_min = z_min
        self.z_max = z_max
        self.r_max = r_max

        self.domain_start = np.array([0.0, self.z_min])
        self.domain_end = np.array([self.r_max, self.z_max])
        self.domain_shape = np.array([self.num_pts_r, self.num_pts_z])    

        # determine voxel size
        self.delta_r = self.r_max / (self.num_pts_r - 1)
        self.delta_z = (self.z_max - self.z_min) / (self.num_pts_z - 1)
        
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
            "z_min": self.z_min,
            "z_max": self.z_max,
            "r_max": self.r_max,
            "num_pts_z": self.num_pts_z,
            "num_pts_r": self.num_pts_r,
            "travel_time_maps": {comp: field.values for comp, field in self.travel_time_fields.items()}
        })
    
    def to_npz(self, outpath, icemodel = 'greenland_simple'):
        """
        Saves calculator metadata and traveltime maps to disk as .npz files.
        """
        r_range = np.linspace(0, self.r_max, self.num_pts_r, dtype = np.float32)
        z_range = np.linspace(self.z_min, self.z_max, self.num_pts_z, dtype = np.float32)

        for comp in self.travel_time_fields:
            np.savez_compressed(
                f'{outpath}_{comp}.npz', 
                r_range_vals = r_range, 
                z_range_vals = z_range,
                data = self.travel_time_fields[comp].values.astype(np.float32),
                antennaz = self.tx_z,
                icemodel = icemodel
                )
    
    def set_ior_and_solve(self, ior, grad_ior, num_big_rays = 0, reflection_at_z = 0.0, early_only = False):
        """
        Supports discontinuous piecewise ice models. Returns `True` when complete.
        
        Parameters
        __________
        ior : function
            Function which returns index of refraction n(z). Should accept floats or ndarrays of z-values (not 3d coordinates).
        grad_ior : function
            Function which returns dn/dz at depth z. See ior.
        num_big_rays : int
            Number of big rays used to calculate refracted maps. Defaults to 0 (in this case no refracted map generated).
        reflection_at_z : float, optional
            z value of air-ice discontinuity. Defaults to z = 0.
        early_only : bool, optional
            Defaults to `False`. If `True`, only first arrivals are generated.
        """

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
            solver.velocity.values = c / 1e9 / iordata

            return solver

        def _get_big_ray(rays, caustic, ray_number, solver):
            """
            Generates big ray node bounds and traveltimes according to current solver domain & mesh size.
            Default ray bounds are generated through reconal raytracer.
            """
            if solver is not None:  # Generate big ray over solver domain
                rvals = solver.traveltime.nodes[:, 0, 0, 0]
            
            else:   # Generate big ray over full domain
                rvals = np.linspace(0, self.r_max, self.num_pts_r)

            # Interpolate raytracer depths, times onto gridded r-values
            bounds = np.array([np.stack((np.interp(rvals, rays[0][ray_number, 0], rays[0][ray_number, 1]),  # First bounding ray (zvals)
                                         np.interp(rvals, rays[0][ray_number, 0], rays[0][ray_number, 2])), axis = 1),  # (traveltimes)
                               np.stack((np.interp(rvals, rays[1][ray_number + 1, 0], rays[1][ray_number + 1, 1]),  # Second bounding ray (zvals)
                                         np.interp(rvals, rays[1][ray_number + 1, 0], rays[1][ray_number + 1, 2])), axis = 1)]) # (traveltimes)
            
            # Sort into top bounding ray, bottom bounding ray
            big_ray = np.array((bounds[np.nanargmin(bounds[..., 0], axis = 0), np.arange(bounds.shape[1])],
                                bounds[np.nanargmax(bounds[..., 0], axis = 0), np.arange(bounds.shape[1])]))
            
            # Interpolate caustic depths, times onto gridded r-values
            if caustic is not None:    
                caustic_vals = np.stack((np.interp(rvals, caustic[0], caustic[1], left = np.nan, right = np.nan),
                                        np.interp(rvals, caustic[0], caustic[2], left = np.nan, right = np.nan)), axis = 1)
                # Replace top ray bound with caustic between contact points
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
        caustic, reflected = ray_utils.get_special_bounds(self.tx_pos, ior, grad_ior, self.r_max, self.z_min, self.z_max, reflection_at_z)

        if caustic is not None:
            rvals = np.linspace(0, self.r_max, self.num_pts_r) 
            caustic_nodes = np.array([rvals, np.interp(rvals, caustic[0], caustic[1], left = np.nan)]).swapaxes(0, 1)
            caustic_nodes = self._coord_to_node(caustic_nodes[np.isfinite(caustic_nodes).all(axis = 1)])
            reflected_nodes = np.array([rvals, np.interp(rvals, reflected[0], reflected[1], left = np.nan, right = np.nan)]).swapaxes(0, 1)
            reflected_nodes = self._coord_to_node(reflected_nodes[np.isfinite(reflected_nodes).all(axis = 1)])

        # Calculate direct rays in the ice
        solver = _get_solver(point = True)
        solver.src_loc = self.tx_pos + [0]
        solver._ntheta = 2

        if caustic is not None:
            _set_boundary_condition(solver, caustic_nodes)

        try:
            solver.known[:, boundary_z_ind + 2:] = True # Prevent solution from propagating into air
        except IndexError:
            pass

        point_solve(solver, src_ind)

        self.travel_time_fields['early'] = solver.traveltime

        # Calculate rays transmitted into the air
        solver = _get_solver()
        solver.traveltime.values[:, boundary_z_ind, :] = self.travel_time_fields['early'].values[:, boundary_z_ind, :]
        solver.unknown[:, boundary_z_ind] = False
        solver.known[:, :boundary_z_ind] = True
        for r_ind in range(self.num_pts_r):
            solver.trial.push(r_ind, boundary_z_ind, 0)
        solver.solve()

        self.travel_time_fields['early'].values[:, boundary_z_ind + 1:] = solver.traveltime.values[:, boundary_z_ind + 1:]

        # Eliminate non-raytracing reflected solutions
        if caustic is not None: # Check that such solutions exist in our domain
            for node in caustic_nodes:
                self.travel_time_fields['early'].values[node[0], node[1]:] = np.inf

        if early_only:  # Save earliest travel-times
            return True

        # Calculate reflected rays: place a line source at the air/ice boundary

        self.travel_time_fields['late'] = pykonal.fields.ScalarField3D(coord_sys = 'cartesian')
        self.travel_time_fields['late'].min_coords = self.domain_start[0], self.domain_start[1], 0
        self.travel_time_fields['late'].npts = self.num_pts_r, boundary_z_ind + 1, 1
        self.travel_time_fields['late'].node_intervals = self.delta_r, self.delta_z, 1

        solver = _get_solver()
        solver.traveltime.values[:, boundary_z_ind] = self.travel_time_fields["early"].values[:, boundary_z_ind]
        try:
            solver.unknown[:, boundary_z_ind] = False
            solver.known[:, boundary_z_ind + 1:] = True
        except IndexError:
            pass
        for r_ind in range(self.num_pts_r):
            solver.trial.push(r_ind, boundary_z_ind, 0)
        solver.solve()

        self.travel_time_fields['late'].values = solver.traveltime.values[:, :boundary_z_ind + 1]
        
        # Eliminate non-raytracing reflected solutions
        if caustic is not None: # Check that such solutions exist in our domain
            for node in reflected_nodes:
                self.travel_time_fields['late'].values[node[0], node[1]:] = np.inf
        
            self.travel_time_fields['late'].values[reflected_nodes[-1, 0]:] = np.inf

        # Calculate refracted rays: big rays method

        if num_big_rays > 0:
            # Raytracer: calculate individual rays & turnover points
            theta_min, theta_max = ray_utils.get_theta_min(self.tx_pos, ior, reflection_at_z) + 0.1, 89.9999 # Exactly 90 degrees would propagate horizontally forever
            mesh = (np.linspace(theta_min, theta_max - 5, num_big_rays + 1), np.linspace(theta_min + 5, theta_max, num_big_rays + 1))
            rays = (ray_utils.get_rays(self.tx_pos, ior, grad_ior, self.r_max, self.z_min, self.z_max, mesh[0], step = self.delta_r, midpoint = True)[0],
                        ray_utils.get_rays(self.tx_pos, ior, grad_ior, self.r_max, self.z_min, self.z_max, mesh[1], step = self.delta_r, midpoint = True)[0])
            S = 4   # node tolerance; ray thickness which triggers adaptive mesh refinement
        
        for iR in range(num_big_rays):
            solver = _get_solver()
            nodes, times = _get_big_ray(rays, caustic, iR, solver)

            # Setting line source
            thindex = np.nonzero(nodes[1, :, 1] - nodes[0, :, 1] < S)[0]
            r1 = thindex[thindex.size - 2 - np.argmin(np.diff(np.flip(thindex)))]   # first r-value where big ray contains a tolerable number of nodes
            solver.traveltime.values[r1, nodes[0, r1, 1] : nodes[1, r1, 1], 0] = self.travel_time_fields['early'].values[r1, nodes[0, r1, 1] : nodes[1, r1, 1], 0]
            solver.known[:r1, nodes[0, r1, 1] : nodes[1, r1, 1], 0] = True
            for z_ind in range(nodes[0, r1, 1], nodes[1, r1, 1]):
                solver.trial.push(r1, z_ind, 0)
            
            _set_boundary_condition(solver, nodes, times)
            solver.solve_first_order()
            _select_relevant_traveltimes(solver, nodes)

            solvers = [solver]  # List to keep all solvers
            big_ray_nodes = [nodes] # List to keep big ray at various resolutions
            big_ray_times = [times] # List to keep raytracing traveltimes

            # Adaptive mesh refinement: divide nodes into quarters where big rays get too thin
            if np.count_nonzero(np.isfinite(solver.traveltime.values[-1, nodes[0, -1, 1] + 1 : nodes[1, -1, 1]])) == 0:
                thindex = thindex[np.logical_and(thindex > r1, thindex < self.num_pts_r - 1)]
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
                    solver.velocity.values = c / 1e9 / iordata

                    # Get big ray
                    nodes, times = _get_big_ray(rays, caustic, iR, solver)
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
                    solver.known[r0] = True
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
            
            self.travel_time_fields['late'].values[~np.isfinite(self.travel_time_fields['late'].values)] = big_ray_map[:, :boundary_z_ind + 1][~np.isfinite(self.travel_time_fields['late'].values)]
        return True

    def get_ind(self, coord):
        return np.transpose(self._coord_to_node(coord))        

    def get_travel_time_map(self, comp = "direct"):
        """
        Returns full traveltime map as numpy array, without metadata.
        """
        return self.travel_time_fields[comp].values

    def get_travel_time(self, coord, comp = "direct", order = "first"):
        """
        Approximates traveltimes at arbitrary coordinates in computational domain.
        First-order approximation is a convenience wrapper around pykonal.fields.ScalarField3D.resample().

        Parameters
        __________
        coord : ndarray (n, 2)
            Coordinates at which to sample traveltime field (order is arbitrary).
        comp : str, optional
            Map component to sample from. Defaults to 'direct'.
        order : str, optional
            Defaults to 'first'. Must be one of the following:
                * 'first': Trilinear interpolation for traveltime at specified coordinate.
                Recommended if arbitrary coordinates are needed.
                * 'zero': Chooses node below and to the left of coordinate and returns
                traveltime at node. Faster at an accuracy loss. Recommended if desired coordinates align with nodes.

        Returns
        _______
        times : ndarray (n,)
            Traveltimes at selected coordinates.
        """
        try:
            if order == 'first':
                coord = np.append(coord, np.zeros((len(coord), 1)), axis = 1)
                return self.travel_time_fields[comp].resample(coord)
                
            elif order == 'zero':
                ind = self.get_ind(coord)
                return self.get_travel_time_ind(ind, comp)
            
            else:
                raise ValueError("Expected 'zero' or 'first' as order argument.")
        
        except KeyError:
            raise KeyError(f"Error: map for component '{comp}' not available!")

    def get_travel_time_ind(self, ind, comp = "direct"):
        """
        Returns traveltime at index position in map.
        """
        return self.travel_time_fields[comp].values[*ind]
    
    def get_tangent_vector(self, coord, comp = "direct", order = "first", unit = False):
        """
        Approximates ray tangent vector at arbitrary coordinates in computational domain using
        traveltime field gradient. First-order approximation uses trilinear interpolation from
        pykonal.fields.VectorField3D.value().

        Parameters
        __________
        coord : ndarray (n, 2)
            Coordinates at which to sample gradient (order is arbitrary).
        comp : str, optional
            Map component to sample from. Defaults to 'direct'.
        order : str, optional
            Defaults to 'first'. Must be one of the following:
                * 'first': Trilinear interpolation for gradient at specified coordinate.
                Recommended if arbitrary coordinates are needed.
                * 'zero': Chooses node below and to the left of coordinate and returns
                gradient at node. Faster at an accuracy loss. Recommended if desired coordinates align with nodes.
        unit : bool, optional
            Defaults to False. If True, returns unit tangent vector.

        Returns
        _______
        grad : ndarray (n, 2)
            Tangent vectors at selected coordinates.
        """
        try:
            if order == 'first':
                coord = np.append(coord, np.zeros((len(coord), 1)), axis = 1)
                if len(coord) == 1:
                    grad = self.travel_time_fields[comp].gradient.value(coord)
                else:
                    grad = np.empty((len(coord), 2), dtype = float)
                    for idx in range(len(coord)):
                        grad[idx] = self.travel_time_fields[comp].gradient.value(coord[idx])[:-1]
                
            elif order == 'zero':
                ind = self.get_ind(coord)
                grad = self.get_gradient_ind(ind, comp)
            
            else:
                raise ValueError("Expected 'zero' or 'first' as order argument.")
            
        except KeyError:
            raise KeyError(f"Error: map for component '{comp}' not available!")

        if unit:
            return grad / np.linalg.norm(grad, axis = 1)[:, np.newaxis]    
        else:
            return grad

    def get_gradient_ind(self, ind, comp = "direct"):
        """
        Returns gradient of traveltime field at index position in map.
        """
        try:
            return self.travel_time_fields[comp].gradient.values[*ind][..., :-1]
        except KeyError:
            raise KeyError(f"Error: map for component '{comp}' not available!")
    
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

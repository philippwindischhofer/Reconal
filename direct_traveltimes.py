import copy
import pykonal
import numpy as np
from . import ray_utils
from scipy.constants import c

class TravelTimeCalculator:

    # All coordinates here are 2d (r, z) in natural feet

    @classmethod
    def FromDict(cls, indict):
        obj = cls(**indict)
        return obj
    
    def __init__(self, tx_z, z_range, r_max, num_pts_z, num_pts_r):

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

    def to_dict(self):        
        return copy.deepcopy({
            "tx_z": self.tx_z,
            "z_range": self.z_range,
            "r_max": self.r_max,
            "num_pts_z": self.num_pts_z,
            "num_pts_r": self.num_pts_r,
            "travel_time_fields": self.travel_time_fields
        })
    
    def set_ior_and_solve(self, ior, grad_ior, num_big_rays, reflection_at_z = 0.0, air_ior = 1.0):

        speed_of_light = c / (1e9) # NuRadio speed of light in m/ns

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
            solver.velocity.values = speed_of_light / iordata
            return solver
        
        def point_solve(solver):
            solver._ntheta = 2
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
            solver.known[np.isfinite(solver.traveltime.values)] = True
            super(pykonal.solver.PointSourceSolver, solver).solve()
        
        def _set_boundary_condition(solver, pixels):
            max_node = solver.traveltime.nodes.shape[:-1]
            boundary_pixels = np.copy(pixels).reshape(-1, pixels.shape[-1])
            boundary_pixels = boundary_pixels[(boundary_pixels < max_node).all(axis = 1)]
            boundary_pixels = boundary_pixels[(boundary_pixels >= 0).all(axis = 1)]
            boundary_inds = tuple(boundary_pixels.swapaxes(0, 1))
            solver.known[*boundary_inds] = True
        
        # Set up ray geometry
        boundary_z_ind = self._coord_to_pykonal([[0, reflection_at_z]])[0][1]
        caustic, turnover_bounds, reflected_bounds = ray_utils.get_special_bounds(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, reflection_at_z)

        rvals = np.arange(0, self.r_max + 1, self.delta_r)
        caustic_bounds = np.array([rvals, np.interp(rvals, caustic[0], caustic[1], left = np.nan)]).swapaxes(0, 1)
        caustic_bounds = caustic_bounds[np.isfinite(caustic_bounds).all(axis = 1)]
        caustic_pixels = self._coord_to_pixel(None, caustic_bounds)

        # Calculate direct rays in the ice
        solver = _get_solver(point = True)
        solver.src_loc = self.tx_pos + [0]

        solver.known[:, boundary_z_ind + 2] = True
        _set_boundary_condition(solver, caustic_pixels)
        point_solve(solver)

        self.travel_time_fields['direct'] = solver.traveltime

        # Calculate rays transmitted into the air: place a line source at the air/ice boundary
        solver = _get_solver()
        solver.traveltime.values[:, boundary_z_ind, :] = self.travel_time_fields["direct"].values[:, boundary_z_ind, :]
        solver.velocity.values[:, boundary_z_ind + 1:] = speed_of_light / air_ior
        solver.unknown[:, boundary_z_ind] = False
        solver.known[:, :boundary_z_ind + 1] = True
        for r_ind in range(self.num_pts_r):
            solver.trial.push(r_ind, boundary_z_ind, 0)
        solver.solve()

        self.travel_time_fields['direct'].values[:, boundary_z_ind + 1:] = solver.traveltime.values[:, boundary_z_ind + 1:]

        # Eliminate refracted solutions from direct map
        turnover_vals = np.interp(np.arange(self.domain_start[1], reflection_at_z, self.delta_z), turnover_bounds[1], turnover_bounds[0], left = np.nan, right = np.nan)
        z_inds = np.nonzero(np.isfinite(turnover_vals))[0]
        turnover_inds = self._coord_to_pixel(None, np.array([turnover_vals[z_inds], np.zeros_like(z_inds, dtype = np.float64)]).swapaxes(0, 1))[:, 0]

        for turnover_ind, z_ind in zip(turnover_inds, z_inds):
            self.travel_time_fields["direct"].values[turnover_ind:, z_ind] = np.inf

    def get_ind(self, coord):
        return np.transpose(self._coord_to_pixel(coord))        
        
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

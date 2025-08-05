import pykonal, copy
import numpy as np
from reconal import ray_utils
from scipy.interpolate import interpn
from time import perf_counter
import matplotlib.pyplot as plt

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

        def _get_solver(iordata):   # Solver over full domain (reflected, first big ray window)
            solver = pykonal.EikonalSolver(coord_sys = "cartesian")
            solver.velocity.min_coords = self.domain_start[0], self.domain_start[1], 0
            solver.velocity.npts = self.num_pts_r, self.num_pts_z, 1
            solver.velocity.node_intervals = self.delta_r, self.delta_z, 1       
            solver.velocity.values = 1.0 / iordata # c = 1 when distance measured in natural feet
            return solver

        def _get_point_solver(iordata): # Point source solver over full domain (direct air, direct ice)
            point_solver = pykonal.solver.PointSourceSolver(coord_sys = "cartesian")
            point_solver.velocity.min_coords = self.domain_start[0], self.domain_start[1], 0
            point_solver.velocity.npts = self.num_pts_r, self.num_pts_z, 1
            point_solver.velocity.node_intervals = self.delta_r, self.delta_z, 1        
            point_solver.velocity.values = 1.0 / iordata
            return point_solver
        
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
                print(self.num_pts_r)
                rvals = np.linspace(0, self.r_max, self.num_pts_r)

            bounds = np.interp(rvals, tracer[0][ray_number, 0], tracer[0][ray_number, 1]), np.interp(rvals, tracer[1][ray_number + 1, 0], tracer[1][ray_number + 1, 1])
            caustic_vals = caustic(rvals)
            big_ray = np.array((np.min(bounds, axis = 0), np.max(bounds, axis = 0)))

            """plt.scatter(rvals, caustic_vals, color = "black", s = 0.3)
            plt.plot(rvals, big_ray[0], color = "red")
            plt.plot(rvals, big_ray[1], color = "blue")"""
            
            if np.isfinite(caustic_vals).any():
                ind1 = np.nanargmin((caustic_vals - bounds[0]) ** 2)
                ind2 = np.nanargmin((caustic_vals - bounds[1]) ** 2) + 1
                big_ray[1][ind1 : ind2] = caustic_vals[ind1 : ind2]

            big_ray = np.swapaxes(np.stack((np.tile(rvals, (2,1)), big_ray), 1), 1, 2)

            pixels = np.array([self._coord_to_pixel(solver, big_ray[0]), self._coord_to_pixel(solver, big_ray[1])])

            return pixels
        
        def _set_soner_condition(solver, pixels):
            max_node = solver.traveltime.nodes.shape[:-1]
            soner_pixels = np.copy(pixels).reshape(-1, pixels.shape[-1]) # Compress since we don't care about max vs min pixels
            print("Yippee")
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
        
        # Build the IOR distribution
        zvals = np.linspace(self.z_range[0], self.z_range[1], self.num_pts_z)
        iorslice = ior(zvals)
        iordata = np.expand_dims(np.tile(iorslice, reps = (self.num_pts_r, 1)), axis = -1)
        
        boundary_z_ind = self._coord_to_pykonal([[0, reflection_at_z]])[0][1]
        caustic = ray_utils.get_caustic(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, reflection_at_z)

        # Calculate rays transmitted into the air
        start = perf_counter()
        solver = _get_point_solver(iordata)
        solver.src_loc = 0, self.tx_z, 0
        solver.solve()
        self.travel_time_maps["direct_air"] = np.copy(solver.traveltime.values)
        self.travel_time_maps["direct_air"][:, :boundary_z_ind, :] = np.nan # this is now unphysical in the ice, as in part of the volume
                                                                            # head-waves will overtake the direct bending modes
        end = perf_counter()
        self.comp_times['direct_air'] = end - start

        # Calculate direct rays in the ice
        start = perf_counter()
        iordata[:, boundary_z_ind:, :] = 10.0 # assign a spuriously large IOR to the air to make sure there are no head waves
                                              # that can overtake the bulk-bending modes that we want
        rvals = np.arange(0, self.r_max + 1, self.delta_r)
        solver = _get_point_solver(iordata)
        caustic_points = np.swapaxes(np.array((rvals, caustic(rvals))), 0, 1)[~np.isnan(caustic(rvals))]
        caustic_pixels = self._coord_to_pixel(solver, caustic_points)
        solver.src_loc = 0, self.tx_z, 0
        
        solver.traveltime.values[:, boundary_z_ind:, :] = np.inf
        solver.known[:, boundary_z_ind:, :] = True
        _set_soner_condition(solver, caustic_pixels)

        solver.solve()
        end = perf_counter()
        self.comp_times['direct_ice'] = end - start

        self.travel_time_maps["direct_ice"] = np.copy(solver.traveltime.values)
        self.travel_time_maps["direct_ice"][:, boundary_z_ind+1:, :] = np.nan # this is now unphysical in the air
        
        # Calculate reflected rays: place a line source at the air/ice boundary
        solver = _get_solver(iordata)
        solver.traveltime.values[:, boundary_z_ind, :] = self.travel_time_maps["direct_ice"][:, boundary_z_ind, :]
        solver.unknown[:, boundary_z_ind, :] = False
        for r_ind in range(self.num_pts_r):
            solver.trial.push(r_ind, boundary_z_ind, 0)
        solver.solve()

        self.travel_time_maps["reflected"] = np.copy(solver.traveltime.values)
        self.travel_time_maps["reflected"][:, boundary_z_ind:, :] = np.nan # this is now unphysical in the air

        end = perf_counter()
        self.comp_times['reflected'] = end - start
        
        # Calculate refracted rays: big rays method

        # Ray tracer: calculate individual rays, turnover points, and intersects (to define caustic)
        theta_min = ray_utils.get_theta_min(self.tx_pos, ior, reflection_at_z)
        theta_max = 87
        mesh = (np.linspace(theta_min + 1, theta_max - 1, num_big_rays + 1), np.linspace(theta_min + 2, theta_max, num_big_rays + 1))
        big_time1_start = perf_counter()
        ray_data = (ray_utils.get_rays(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, mesh[0]), ray_utils.get_rays(self.tx_pos, ior, grad_ior, self.r_max, self.z_range, mesh[1]))
        big_time1_end = perf_counter()

        big_time1 = big_time1_end - big_time1_start
        print("Ray-tracing time:", big_time1)
        print("Generating caustic")
        caustic = ray_utils.get_caustic(self.tx_pos, ior, grad_ior, self.r_max, self.z_range,  reflection_at_z)
        tracer = ray_data[0][0], ray_data[1][0]
        turnover = ray_data[0][1]
        tolerance = 6 # pixel tolerance; how small a big ray can get before adaptive gridsize kicks in

        self.travel_time_maps["refracted"] = np.full((self.num_pts_r, self.num_pts_z, 1), np.nan)
        
        for i in range(num_big_rays):

            if i == 1:
                break

            print("Generating traveltime field for big ray " + str(i + 1) + " of " + str(num_big_rays))
            
            solver = _get_solver(iordata)
            
            pixels = _get_big_ray(tracer, caustic, i, solver)


            r1 = self._coord_to_pixel(solver, np.array([turnover[i]]))[0, 0]    # first ray turns over

            # Setting line source
            solver.traveltime.values[:r1 + 1, :, 0] = np.copy(self.travel_time_maps['direct_ice'])[:r1 + 1, :, 0]
            solver.known[:r1 + 1, pixels[0, r1, 1] : pixels[1, r1, 1], 0] = True
            for z_ind in range(pixels[0, r1, 1], pixels[1, r1, 1]):
                solver.trial.push(r1 + 1, z_ind, 0)

            _set_soner_condition(solver, pixels)
            solver.solve()
            _select_relevant_traveltimes(solver, pixels)


            z0 = solver.vv.min_coords[1]
            dr = solver.vv.node_intervals[0]
            dz = solver.vv.node_intervals[1]
            """fig1, ax1 = plt.subplots()
            ax1.scatter(pixels[1, :, 0] * dr, pixels[1, :, 1] * dz + z0, s = 0.3, color = "black", zorder = 1)
            ax1.scatter(pixels[0, :, 0] * dr, pixels[0, :, 1] * dz + z0, s = 0.3, color = "black", zorder = 1)
            ax1.set_xlim(0, 7500)
            ax1.set_ylim(-4500, 0)
            ax1.set_title("1 big ray (2 normal rays, caustic)")"""


            # Adaptive grid-sizing: divide pixels into quarters where rays get too thin (rewrite this comment when ur brain is not leaking out of ur ears)
            thindex = np.array(np.nonzero(pixels[1, :,1] - pixels[0, :,1] <= tolerance))
            thindex = thindex[thindex >= r1]

            # Defining some lists to keep track of things
            solvers = [solver]
            maps = [np.copy(solver.tt.values)]
            extents = [[solver.tt.min_coords[0], solver.tt.max_coords[0], solver.tt.min_coords[1], solver.tt.max_coords[1]]]
            x = 1

            while thindex.size > 0:
                print("zoom in " + str(x))

                r0 = np.min(thindex)    # first point after turnover where ray thickness is below our tolerance
                rf = np.max(thindex)    # after this point, ray thickness is above our tolerance; return to normal grid size

                solver = _get_updated_solver(solvers[-1], 2, r0, rf, old_pixels = pixels)

                # Get big ray
                pixels = _get_big_ray(tracer, caustic, i, solver)

                # Set line source
                nodes = solver.traveltime.nodes[0, ...]
                solver.traveltime.values[0, :, 0] = solvers[-1].tt.resample(nodes.reshape(-1, 3))
                solver.known[0, :, 0] = True
                for z_ind in range(pixels[0, 0, 1], pixels[1, 0, 1]):
                    solver.trial.push(0, z_ind, 0)

                _set_soner_condition(solver, pixels)

                solver.solve()

                _select_relevant_traveltimes(solver, pixels)

                solvers.append(solver)
                big_ray_map = np.copy(solver.traveltime.values)
                maps.append(big_ray_map)
                extents.append([solver.tt.min_coords[0], solver.tt.max_coords[0], solver.tt.min_coords[1], solver.tt.max_coords[1]])
                
                """
                fig, ax = plt.subplots()
                tvals = np.flip(np.transpose(big_ray_map, axes = (1, 0, 2)), axis = 0)
                image = ax.imshow(tvals, cmap = "rainbow", extent = [solver.tt.min_coords[0], solver.tt.max_coords[0], solver.tt.min_coords[1], solver.tt.max_coords[1]])

                z0 = solver.vv.min_coords[1]
                dr = solver.vv.node_intervals[0]
                dz = solver.vv.node_intervals[1]
                r0_val = solver.vv.nodes[0, 0, 0, 0]
                ax.scatter(pixels[1, :, 0] * dr + r0_val, pixels[1, :, 1] * dz + z0, color = "black", s = 3)
                ax.scatter(pixels[0, :, 0] * dr + r0_val, pixels[0, :, 1] * dz + z0, color = "black", s = 3)
                ax.vlines((solver.tt.min_coords[0], solver.tt.max_coords[0]), self.domain_start[1], self.domain_end[1], lw = 0.3)"""

                thindex = np.array(np.nonzero(pixels[1, :, 1] - pixels[0, :, 1] <= tolerance))

                if np.any(np.isfinite(solver.tt.values[-1, ...])):
                    break

                x += 1

            # Return to normal grid size
            # Basically what we've done at this point is build a bunch of windows centering on the thinnest part of the ray.
            # We've propagated to the end of the smallest window, so now we propagate end to end until we're on the largest window (our full domain).
            # I will draw a picture of this

            solvers_iterate = list.copy(solvers)
            solvers_iterate.reverse()
            for ii in range(1, len(solvers_iterate)):
                print("zoom out " + str(ii))
                
                r0 = self._coord_to_pixel(solvers_iterate[ii], [solvers_iterate[ii - 1].vv.max_coords[:-1]])[0, 0]
                rf = solvers_iterate[ii].vv.nodes.shape[0]
                z0 = 0
                zf = solvers_iterate[ii].vv.nodes.shape[1]
                
                solver = _get_updated_solver(solvers_iterate[ii], 1, r0, rf, z0, zf)
                """
                solver = pykonal.EikonalSolver(coord_sys = "cartesian") # Define new solver between endpoints of two solvers in list
                solver.velocity.min_coords = r0_val, self.domain_start[1], 0
                solver.velocity.node_intervals = solvers_iterate[ii - 1].velocity.node_intervals[0] * 2, solvers_iterate[ii - 1].velocity.node_intervals[1] * 2, 1  # Step gridsize up by 4 (2 in r direction, 2 in z direction)
                solver.velocity.npts = int((rf_val - r0_val) / solver.velocity.node_intervals[0]), (1/2) * solvers_iterate[ii - 1].velocity.npts[1], 1
                
                zvals = np.linspace(self.z_range[0], self.z_range[1], solver.velocity.npts[1])
                iorslice = ior(zvals)
                iordata = np.expand_dims(np.tile(iorslice, reps = (solver.velocity.npts[0], 1)), axis = -1)
                veldata = 1.0 / iordata

                solver.velocity.values = veldata"""

                # Get big ray
                pixels = _get_big_ray(tracer, caustic, i, solver)

                # Set line source
                min_z_node = self._coord_to_pixel(solver, np.array([solvers[-1].tt.min_coords[:-1]]))[0, 1]
                max_z_node = self._coord_to_pixel(solver, np.array([solvers[-1].tt.max_coords[:-1]]))[0, 1]
                if max_z_node - min_z_node != solvers[-1].traveltime.values.shape[1] // 2: # Correct for rounding in coord to pixel function
                    solver.traveltime.values[0, min_z_node : max_z_node, 0] = solvers[-1].traveltime.values[-1, :-2:2, 0]
                else:
                    solver.traveltime.values[0, min_z_node : max_z_node, 0] = solvers[-1].traveltime.values[-1, ::2, 0]
                solver.known[0, :, 0] = True
                for z_ind in range(pixels[0, 0, 1], pixels[1, 0, 1]):
                    solver.trial.push(0, z_ind, 0)

                _set_soner_condition(solver, pixels)

                solver.solve()

                _select_relevant_traveltimes(solver, pixels)

                solvers.append(solver)
                big_ray_map = np.copy(solver.traveltime.values)
                maps.append(big_ray_map)
                extents.append([solver.tt.min_coords[0], solver.tt.max_coords[0], solver.tt.min_coords[1], solver.tt.max_coords[1]])

                
                """fig, ax = plt.subplots()
                tvals = np.flip(np.transpose(big_ray_map, axes = (1, 0, 2)), axis = 0)
                image = ax.imshow(tvals, cmap = "rainbow", extent = [solver.tt.min_coords[0], solver.tt.max_coords[0], solver.tt.min_coords[1], solver.tt.max_coords[1]])

                z0 = solver.vv.min_coords[1]
                dr = solver.vv.node_intervals[0]
                dz = solver.vv.node_intervals[1]
                ax.scatter(pixels[1, :, 0] * dr + r0_val, pixels[1, :, 1] * dz + z0, color = "black", s = 3)
                ax.scatter(pixels[0, :, 0] * dr + r0_val, pixels[0, :, 1] * dz + z0, color = "black", s = 3)
                ax.vlines((solver.tt.min_coords[0], solver.tt.max_coords[0]), self.domain_start[1], self.domain_end[1], lw = 0.3)"""

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

            fig, ax = plt.subplots()
            axins = ax.inset_axes(bounds = [0.1, 0.15, 0.4, 0.4], xlim = (675, 975), ylim = (-300, -125))  
            box, connectors = ax.indicate_inset_zoom(axins, edgecolor = 'black')
            for map, extent in zip(maps, extents):
                tvals = np.flip(np.transpose(map, axes = (1, 0, 2)), axis = 0)
                image = ax.imshow(tvals, cmap = "rainbow", extent = extent)
                imin = axins.imshow(tvals, cmap = 'rainbow', extent = extent)
                image.set_clim(0, 6300)
                imin.set_clim(0, 6300)
            # ax.plot(pixels[1, :, 0] * dr, pixels[1, :, 1] * dz + z0, color = "black", zorder = 1)
            # ax.plot(pixels[0, :, 0] * dr, pixels[0, :, 1] * dz + z0, color = "black", zorder = 1)
            ax.set_xlim(0, self.r_max)
            ax.set_ylim(self.z_range[0], self.z_range[1])
            ax.set_title("Big ray " +  str(i + 1) + " of " + str(num_big_rays) + " (with adaptive gridsizing)")
            ax.set_xlabel("Distance from source (m)")
            ax.set_ylabel("Depth (m)")
            cbar = fig.colorbar(image)
            cbar.set_label("Traveltime (ns)")
            plt.show()
            exit()
        self.travel_time_maps['refracted'][self.travel_time_maps['refracted'] < self.travel_time_maps['direct_ice'] + 1] = np.nan
        end = perf_counter()
        self.comp_times['refracted'] = end - start
        

        # self.travel_time_maps['refracted'][self.travel_time_maps['refracted'] < self.travel_time_maps['direct_ice'] + 1] = np.nan

        fig, ax = plt.subplots()
        tvals = np.flip(np.transpose(self.travel_time_maps['refracted'], axes = (1, 0, 2)), axis = 0)
        image = ax.imshow(tvals, cmap = "rainbow", extent = [self.domain_start[0], self.domain_end[0], self.domain_start[1], self.domain_end[1]])
        image.set_clim(0, 6300)
        ax.set_xlim(0, self.r_max)
        ax.set_ylim(self.z_range[0], self.z_range[1])
        ax.set_title(str(num_big_rays) + " big rays (with adaptive gridsizing)")
        ax.set_xlabel("Distance from source (m)")
        ax.set_ylabel("Depth (m)")
        cbar = fig.colorbar(image)
        cbar.set_label("Traveltime (ns)")
        # fig.savefig("/Users/mcb/Desktop/work/big ray tracing/adaptive gridsizing/Big ray " +  str(i + 1) + " of " + str(num_big_rays) + ".pdf", dpi = 1000)

        # big_ray_map[big_ray_map <= self.travel_time_maps['direct_ice'] + 1] = np.nan # Eliminate first arrivals

        # fig, ax = plt.subplots()
        # tvals = np.flip(np.transpose(big_ray_map, axes = (1, 0, 2)), axis = 0)
        # image = ax.imshow(tvals, cmap = "rainbow", extent = [solver.tt.min_coords[0], solver.tt.max_coords[0], self.domain_start[1], self.domain_end[1]])
        # image = ax.imshow(tvals, cmap = "rainbow", extent = [new_solver.tt.min_coords[0], new_solver.tt.max_coords[0], self.domain_start[1], self.domain_end[1]])
        # image.set_clim(0, 9000)
        # ax.plot(np.linspace(0, self.r_max, self.num_pts_r), big_ray[0], color = "red", lw = 0.3)
        # ax.plot(np.linspace(0, self.r_max, self.num_pts_r), big_ray[1], color = "red", lw = 0.3)
        # ax.plot(np.linspace(new_solver.tt.min_coords[0], new_solver.tt.max_coords[0], new_solver.tt.npts[0]), ray[0], color = "black", lw = 0.3)
        # ax.plot(np.linspace(new_solver.tt.min_coords[0], new_solver.tt.max_coords[0], new_solver.tt.npts[0]), ray[1], color = "black", lw = 0.3)

        # r0 = solver.vv.min_coords[0]
        # rf = solver.vv.max_coords[0]
        """z0 = new_solver.vv.min_coords[1]
        zf = new_solver.vv.max_coords[1]
        dr = new_solver.vv.node_intervals[0]
        dz = new_solver.vv.node_intervals[1]"""
        # ax.scatter(max_pixels[:, 0] * dr + r0, max_pixels[:, 1] * dz + z0, s = 1)
        # ax.scatter(min_pixels[:, 0] * dr + r0, min_pixels[:, 1] * dz + z0, s = 1)
        """ax.set_xlim(0, 4000)
        ax.set_ylim(-2000, 0)
        # ax.vlines((r1, new_solver.tt.min_coords[0], new_solver.tt.max_coords[0]), self.domain_start[1], self.domain_end[1], lw = 0.3)
        # ax.set_xlim(solver_fine.tt.min_coords[0], solver_fine.tt.max_coords[0])
        cbar = fig.colorbar(image,fraction=0.046, pad=0.04)
        cbar.set_label("Traveltime (ns)")
        ax.set_xlabel("Distance from source (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Refracted travel-time map inside one big ray")
        fig.savefig(("/Users/mcb/Desktop/work/big ray tracing/direct to turnover/help " + str(i) + ".pdf"), dpi = 1000)"""

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


"""if i == 10:
    tt = solver.near_field.traveltime.values
    help = np.swapaxes(np.argwhere(~(tt == np.inf)), 0, 1)

    fig = plt.figure()
    ax = fig.add_subplot()
    image = ax.scatter(help[0], help[2], c = tt[help[0], 0, help[2]])
    fig.colorbar(image)
    image.set_clim(0, 20)
    plt.show()
    exit()"""

"""for pixel in near_max_sph_pixel:
        if np.any(pixel < solver.near_field.velocity.min_coords) or np.any(pixel > solver.near_field.velocity.max_coords):
            continue
        solver.near_field.traveltime.values[pixel[0], pixel[1], pixel[2]:] = 1e10
        solver.near_field.known[pixel[0], pixel[1], pixel[2]:] = True

for pixel in near_min_sph_pixel:
    if np.any(pixel < solver.near_field.velocity.min_coords) or np.any(pixel > solver.near_field.velocity.max_coords):
        continue
    solver.near_field.traveltime.values[pixel[0], pixel[1], :pixel[2]] = 1e10
    solver.near_field.known[pixel[0], pixel[1], :pixel[2]] = True"""

"""near_min_xy = np.append(near_min_xy, np.zeros((np.shape(near_min_xy)[0], 1)), axis=1)
near_max_xy = np.append(near_max_xy, np.zeros((np.shape(near_max_xy)[0], 1)), axis=1)

near_min_sph = pykonal.transformations.xyz2sph(near_min_xy, origin = (self.tx_pos + [0.0]))
near_max_sph = pykonal.transformations.xyz2sph(near_max_xy, origin = (self.tx_pos + [0.0]))
near_min_sph_pixel = self._coord_to_pixel(near_min_sph, sph = True, solver = solver)
near_max_sph_pixel = self._coord_to_pixel(near_max_sph, sph = True, solver = solver)"""

# print(near_sph_pixel[...,2]) # Seeing the range of phi values
# print(near_sph_pixel)

"""fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
r = near_sph[...,0]
theta = near_sph[..., 2]
ax.plot(theta, r)
plt.show()

exit()"""


"""
Everything below here is point source stuff & therefore does not work :(


def _get_big_rays(theta_min, theta_max, src, num, caustic, nrho, drho):
    
    far_field = np.full((num, 2, self.num_pts_r), np.nan) # big ray number, (lower bound, upper bound), r value
    near_field = np.full((num, 2, nrho), np.nan) # Same as far_field, extending only to end of spherical grid

    if caustic == True:
        mesh = (np.linspace(theta_min + 1, theta_max - 1, num + 1), np.linspace(theta_min + 2, theta_max, num + 1))
        tracer = (ray_utils.get_rays(src, ior, grad_ior, self.r_max, self.z_range, self.delta_r, 0.001, mesh[0])[0], ray_utils.get_rays(src, ior, grad_ior, self.r_max, self.z_range, self.delta_r, 0.001, mesh[1])[0])

        bounds = np.full((2, num + 1, self.num_pts_r), np.nan)
        near_bounds = np.full((2, num + 1, nrho), np.nan)

        for i in range(num + 1):
            bounds[:, i] = np.interp(np.linspace(0, self.r_max, self.num_pts_r), tracer[0][i, 0], tracer[0][i, 1]), np.interp(np.linspace(0, self.r_max, self.num_pts_r), tracer[1][i, 0], tracer[1][i, 1])
            near_bounds[:, i] = np.interp(np.linspace(0, nrho * drho, nrho), tracer[0][i, 0], tracer[0][i, 1]), np.interp(np.linspace(0, nrho * drho, nrho), tracer[1][i, 0], tracer[1][i, 1])
        
        for i in range(num):

            rays = bounds[0][i], bounds[1][i + 1]
            far_field[i] = np.minimum(*rays), np.maximum(*rays)

            intersects = np.array((ray_utils.get_intersect((mesh[0][i], mesh[0][i] + 1), src, ior, grad_ior, self.r_max, self.delta_r, 0.0001, reflection_at_z), ray_utils.get_intersect((mesh[1][i + 1], mesh[1][i + 1] + 1), src, ior, grad_ior, self.r_max, self.delta_r, 0.0001, reflection_at_z)), dtype = int)
            caustic_points = np.interp(np.arange(intersects[0,0], intersects[1,0], self.delta_r), intersects[:,0], intersects[:,1])
            far_field[i, 1, intersects[0,0]:intersects[1,0]] = caustic_points

            near_rays = near_bounds[0][i], near_bounds[1][i + 1]
            near_field[i] = np.minimum(*near_rays), np.maximum(*near_rays)

    else:
        mesh = np.linspace(theta_min, theta_max, num + 2)
        tracer, turnover = ray_utils.get_rays(src, ior, grad_ior, self.r_max, self.z_range, self.delta_r, 0.001, mesh)

        bounds = np.full((num + 2, self.num_pts_r), np.nan)
        near_bounds = np.full((num + 2, nrho), np.nan)
        for i, ray in enumerate(tracer):
            bounds[i] = np.interp(np.linspace(0, self.r_max, self.num_pts_r), ray[0], ray[1])
            near_bounds[i] = np.interp(np.linspace(0, nrho * drho, nrho), ray[0], ray[1])
        
        for i in range(1, num + 1):
            far_rays = bounds[i - 1], bounds[i], bounds[i + 1]
            far_field[i - 1] = np.minimum.reduce(far_rays), np.maximum.reduce(far_rays)

            near_rays = near_bounds[i - 1], near_bounds[i], near_bounds[i + 1]
            near_field[i - 1] = np.minimum.reduce(near_rays), np.maximum.reduce(near_rays)

    return near_field, far_field, turnover

# Setting grid parameters
theta_min = ray_utils.get_theta_min(self.tx_pos, ior, reflection_at_z)

if caustic == True:
    theta_max = 87
else:
    theta_max = 89.99

phi_range = np.radians(theta_max - theta_min)
dphi = (1 / (8 * num_big_rays)) * np.radians(phi_range)
drho = (1 / 8) * min(self.delta_r, self.delta_z)
nrho = int(40 * max(self.delta_r, self.delta_z) / (np.sin(phi_range / (2 * num_big_rays)) * drho))
print(nrho)

near_field, far_field = _get_big_rays(theta_min, theta_max, self.tx_pos, num_big_rays, caustic, nrho, drho)

self.travel_time_maps["refracted"] = np.full((self.num_pts_r, self.num_pts_z, 1), np.nan)

for i in range(num_big_rays):
    
    if i != 0:
        break
    
    near_ray = near_field[i]
    far_ray = far_field[i]

    print("Generating traveltime field for ray " + str(i))

    solver = _get_point_solver(iordata, nrho, drho, dphi)
    solver.src_loc = self._coord_to_pykonal([self.tx_pos])[0]
    
    solver.initialize_near_field_grid()
    solver.interpolate_far_field_velocity_onto_near_field()

    far_min_pixels = self._coord_to_pixel(np.swapaxes(np.stack((np.linspace(0, self.r_max, self.num_pts_r), far_ray[0])), 0, 1))
    far_max_pixels = self._coord_to_pixel(np.swapaxes(np.stack((np.linspace(0, self.r_max, self.num_pts_r), far_ray[1])), 0, 1))
    far_pixels = np.vstack((far_min_pixels, far_max_pixels))

    near_min_xy = np.swapaxes(np.stack((np.linspace(0, solver.nrho * solver.drho, solver.nrho), near_ray[0])), 0, 1)
    near_max_xy = np.swapaxes(np.stack((np.linspace(0, solver.nrho * solver.drho, solver.nrho), near_ray[1])), 0, 1)
    near_xy = np.vstack((near_min_xy, near_max_xy))
    near_xy = np.append(near_xy, np.zeros((np.shape(near_xy)[0], 1)), axis=1)
    near_sph = pykonal.transformations.xyz2sph(near_xy, origin = (self.tx_pos + [0.0]))
    near_sph_pixel = self._coord_to_pixel(near_sph, sph = True, solver = solver)
    near_sph_pixel[..., 1] = 0

    # Setting Soner condition in near field
    phi_lower, phi_upper = near_sph_pixel[1, 2], near_sph_pixel[solver.nrho + 1, 2]
    max_node = np.shape(solver.near_field.velocity.nodes)[:-1]
    for pixel in near_sph_pixel:
        if np.any(pixel < 0) or np.any(pixel >= max_node):
            print("hey")
            continue
        solver.near_field.traveltime.values[*pixel] = 1e10
        solver.near_field.known[*pixel] = True

    # Initializing narrow band of near field
    for ip in range(phi_lower - 200, phi_upper + 3):
        idx = (0, 0, ip)
        solver.near_field.traveltime.values[idx] = solver.drho / solver.near_field.velocity.values[idx]
        solver.near_field.unknown[idx] = False
        solver.near_field.trial.push(*idx)

    # Solving near field and transitioning to far field
    solver.near_field.solve()
    big_ray_map = np.copy(solver.near_field.traveltime.values)

    # What is wrong
    nodes = solver.near_field.traveltime.nodes
    xx = nodes[..., 0] * np.cos(nodes[..., 1]) * np.cos(nodes[...,2])
    yy = nodes[..., 0] * np.cos(nodes[..., 1]) * np.sin(nodes[...,2])

    phi_upper = phi_upper * dphi
    phi_lower = phi_lower * dphi

    fig, ax = plt.subplots()
    qmesh = ax.pcolormesh(
        xx[:,0,:],
        yy[:,0,:],
        big_ray_map[:,0,:],
        cmap = plt.get_cmap('rainbow'),
        zorder = 10
    )
    fig.colorbar(qmesh)
    qmesh.set_clim((5e9, 1e10))
    ax.plot(np.linspace(0, self.r_max, self.num_pts_r), far_field[i, 0] + 200, color = "black")
    ax.plot(np.linspace(0, self.r_max, self.num_pts_r), far_field[i, 1] + 200, color = "black")
    ax.plot(np.linspace(0, self.r_max, self.num_pts_r), np.linspace(0, self.r_max, self.num_pts_r) * np.tan(phi_lower), color = "red")
    ax.plot(np.linspace(0, self.r_max, self.num_pts_r), np.linspace(0, self.r_max, self.num_pts_r) * np.tan(phi_upper), color = "red")
    # ax.set_xlim(-0.05, 0.20)
    # ax.set_ylim(-0.05, 0.20)
    plt.show()


    solver.interpolate_near_field_traveltime_onto_far_field()

    # Setting Soner condition in Cartesian coordinates
    for pt in far_pixels:
        if np.any(pt < solver.velocity.min_coords) or np.any(pt > solver.velocity.max_coords):
            continue
        solver.traveltime.values[*pt] = 1e10
        solver.known[*pt] = True
    
    big_ray_map = np.copy(solver.traveltime.values)

    # big_ray_map[np.isinf(big_ray_map)] = np.nan # Eliminate boundary conditions
    big_ray_map[big_ray_map <= self.travel_time_maps['direct_ice'] + 1] = np.nan # Eliminate first arrivals

    fig, ax = plt.subplots()
    print(big_ray_map)
    tvals = np.flip(np.transpose(big_ray_map, axes = (1, 0, 2)), axis = 0)
    image = ax.imshow(tvals, cmap = "rainbow", extent = [self.domain_start[0], self.domain_end[0], self.domain_start[1], self.domain_end[1]])
    image.set_clim(0, 9000)
    ax.plot(np.linspace(0, self.r_max, self.num_pts_r), far_field[i, 0], color = "black", lw = 0.3)
    ax.plot(np.linspace(0, self.r_max, self.num_pts_r), far_field[i, 1], color = "black", lw = 0.3)
    ax.set_xlim(0, solver.nrho * solver.drho)
    ax.set_ylim(self.tx_z - solver.nrho * solver.drho, self.tx_z + solver.nrho * solver.drho)
    fig.colorbar(image)
    fig.savefig(("/Users/mcb/Desktop/work/big ray tracing/near field " + str(i + 1) + " of " + str(num_big_rays) + ".pdf"), dpi = 300)
    plt.show()

    solver.initialize_far_field_narrow_band()
    super(pykonal.solver.PointSourceSolver, solver).solve()
    big_ray_map = np.copy(solver.traveltime.values)

    # Selecting relevant traveltimes
    for r in range(self.num_pts_r):
        min_z = far_min_pixels[r, 1]
        max_z = far_max_pixels[r, 1]
        if min_z < 0:
            min_z = 0
        solver.traveltime.values[r, :min_z + 1, 0] = np.nan
        solver.traveltime.values[r, max_z - 1:, 0] = np.nan # Unphysical outside ray boundaries

    # Selecting relevant traveltimes
    big_ray_map[np.isinf(big_ray_map)] = np.nan # Eliminate boundary conditions
    # big_ray_map[big_ray_map <= self.travel_time_maps['direct_ice'] + 1] = np.nan # Eliminate first arrivals

    fig, ax = plt.subplots()
    tvals = np.flip(np.transpose(big_ray_map, axes = (1, 0, 2)), axis = 0)
    image = ax.imshow(tvals, cmap = "rainbow", extent = [self.domain_start[0], self.domain_end[0], self.domain_start[1], self.domain_end[1]])
    image.set_clim(0, 9000)
    ax.plot(np.linspace(0, self.r_max, self.num_pts_r), far_field[i, 0], color = "black", lw = 0.3)
    ax.plot(np.linspace(0, self.r_max, self.num_pts_r), far_field[i, 1], color = "black", lw = 0.3)
    ax.set_xlim(0, self.domain_end[0])
    ax.set_ylim(self.domain_start[1], self.domain_end[1])
    fig.colorbar(image)
    fig.savefig(("/Users/mcb/Desktop/work/big ray tracing/big ray " + str(i + 1) + " of " + str(num_big_rays) + ".pdf"), dpi = 300)
    plt.show()

    # big_ray_map[abs(big_ray_map - self.travel_time_maps['refracted']) < 0.1] = np.nan # Eliminate duplicates from overlapping big rays
    self.travel_time_maps["refracted"][~np.isnan(big_ray_map)] = big_ray_map[~np.isnan(big_ray_map)]

fig = plt.figure()
ax = fig.add_subplot()
tvals = self.travel_time_maps['refracted']
tvals = np.flip(np.transpose(tvals, axes = (1, 0, 2)), axis = 0)
image = ax.imshow(tvals, cmap = "rainbow", extent = [self.domain_start[0], self.domain_end[0], self.domain_start[1], self.domain_end[1]])
image.set_clim(0, 9000)
# ax.set_ylim(-210, -190)
# ax.set_xlim(0, 10)
fig.colorbar(image)

if caustic == True:
    ax.set_title(("Refracted traveltime map, nrays = " +  str(num_big_rays) + ", \n with caustic method"))
else:
    ax.set_title(("Refracted traveltime map, nrays = " +  str(num_big_rays)))

ax.set_ylabel("Depth (m)")
ax.set_xlabel("Range (m)")
# ax.scatter(0, self.tx_z, c = "black")
# plt.show()
# fig.savefig(("nrays" + str(num_big_rays) + " caustic" + str(caustic)))


def _coord_to_pykonal(self, coord, sph = False, solver = None):
        return tuple(self._coord_to_pixel(coord, sph, solver))
        
def _coord_to_pixel(self, coord, sph = False, solver = None):
    if sph == True:
        return self._coord_to_frac_sph_pixel(coord, solver).astype(int)
    else:
        return self._coord_to_frac_pixel(coord).astype(int)
    
def _coord_to_frac_pixel(self, coord):        

    if isinstance(coord, list):
        coord = np.array(coord)
        
    pixel_2d = (coord - self.domain_start) / (self.domain_end - self.domain_start) * self.domain_shape
    pixel_3d = np.append(pixel_2d, np.zeros((len(coord), 1)), axis = 1)
    return pixel_3d

def _coord_to_frac_sph_pixel(self, coord, solver):
    if isinstance(coord, list):
        coord = np.array(coord)
    return coord / np.array([solver.drho, solver.dtheta, solver.dphi])



# Non-caustics functionality (big ray = envelope of 3 little rays)
else:
    theta_max = 89.99
    mesh = np.linspace(theta_min, theta_max, num + 2)
    ray_data = ray_utils.get_rays(src, ior, grad_ior, self.r_max, self.z_range, self.delta_r, 0.001, mesh)
    tracer = ray_data[0]
    turnover = ray_data[1][:num]

    bounds = np.full((num + 2, self.num_pts_r), np.nan)

    for i, ray in enumerate(tracer):
        bounds[i] = np.interp(np.linspace(0, self.r_max, self.num_pts_r), ray[0], ray[1])

    for i in range(1, num + 1):
        rays = bounds[i - 1], bounds[i], bounds[i + 1]
        big_rays[i - 1] = np.minimum.reduce(rays), np.maximum.reduce(rays)    

"""

"""
# Big rays back when you could get them all at once
def _get_big_rays(src, num):
    theta_min = ray_utils.get_theta_min(src, ior, reflection_at_z)
    theta_max = 87
    big_rays = np.full((num, 2, self.num_pts_r), np.nan) # big ray number, (lower bound, upper bound), r value
    mesh = (np.linspace(theta_min + 1, theta_max - 1, num + 1), np.linspace(theta_min + 2, theta_max, num + 1))
    ray_data = (ray_utils.get_rays(src, ior, grad_ior, self.r_max, self.z_range, self.delta_r, 0.001, mesh[0]), ray_utils.get_rays(src, ior, grad_ior, self.r_max, self.z_range, self.delta_r, 0.001, mesh[1]))
    tracer = ray_data[0][0], ray_data[1][0]
    turnover = ray_data[0][1]

    bounds = np.full((2, num + 1, self.num_pts_r), np.nan)

    for i in range(num + 1):
        bounds[:, i] = np.interp(np.linspace(0, self.r_max, self.num_pts_r), tracer[0][i, 0], tracer[0][i, 1]), np.interp(np.linspace(0, self.r_max, self.num_pts_r), tracer[1][i, 0], tracer[1][i, 1])
    
    for i in range(num):
        rays = bounds[0][i], bounds[1][i + 1]
        big_rays[i] = np.minimum(*rays), np.maximum(*rays)

        intersects = np.array((ray_utils.get_intersect((mesh[0][i], mesh[0][i] + 1), src, ior, grad_ior, self.r_max, self.delta_r, 0.0001, reflection_at_z), ray_utils.get_intersect((mesh[1][i + 1], mesh[1][i + 1] + 1), src, ior, grad_ior, self.r_max, self.delta_r, 0.0001, reflection_at_z)), dtype = int)
        caustic_points = np.interp(np.arange(intersects[0,0], intersects[1,0], self.delta_r), intersects[:,0], intersects[:,1])
        big_rays[i, 1, intersects[0,0]:intersects[1,0]] = caustic_points

    return big_rays, turnover, intersects
"""
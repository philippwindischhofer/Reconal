import numpy as np
import defs

def show_map(ax, intmap, axis_a, axis_b, cmap = "bwr", aspect = "auto", vmin = None, vmax = None, fs = 13):

    intmap_to_plot = np.squeeze(intmap["map"])
    
    if vmin is None:
        vmin = np.min(intmap_to_plot)

    if vmax is None:
        vmax = np.max(intmap_to_plot)

    cscale = max(abs(vmin), abs(vmax))
    vmin, vmax = -cscale, cscale

    im = ax.imshow(np.flip(np.transpose(intmap_to_plot), axis = 0),
                   extent = [intmap[axis_a][0] * defs.cvac, intmap[axis_a][-1] * defs.cvac,
                             intmap[axis_b][0] * defs.cvac, intmap[axis_b][-1] * defs.cvac],
                   cmap = cmap, vmax = vmax, vmin = vmin, aspect = aspect)
    
    ax.set_xlabel(f"{axis_a} [m]", fontsize = fs)
    ax.set_ylabel(f"{axis_b} [m]", fontsize = fs)
    ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
    ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)

    return im

def show_map_ang(ax, intmap, axis_a, axis_b, cmap = "bwr", aspect = "auto", vmin = None, vmax = None, fs = 13, symcolor = True):

    intmap_to_plot = np.squeeze(intmap["map"])
    
    if vmin is None:
        vmin = np.min(intmap_to_plot)

    if vmax is None:
        vmax = np.max(intmap_to_plot)

    if symcolor:
        cscale = max(abs(vmin), abs(vmax))
        vmin, vmax = -cscale, cscale

    im = ax.imshow(np.flip(np.transpose(intmap_to_plot), axis = 0),
                   extent = [intmap[axis_a][0], intmap[axis_a][-1],
                             intmap[axis_b][0], intmap[axis_b][-1]],
                   cmap = cmap, vmax = vmax, vmin = vmin, aspect = aspect)
    
    ax.set_xlabel(f"{axis_a}", fontsize = fs)
    ax.set_ylabel(f"{axis_b}", fontsize = fs)
    ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
    ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)

    return im

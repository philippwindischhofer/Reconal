import numpy as np
import defs

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def make_direction_plot(outpath, elevations, azimuths, obs_label = "", plot_title = "", labels = [], 
                        show_labels = True, fs = 13, cmap = "plasma", inner_label = "", fitline = None):

    fig = plt.figure(figsize = (6, 6), layout = "constrained")
    gs = GridSpec(1, 1, figure = fig)
    ax = fig.add_subplot(gs[0])

    neg_azimuths = -np.array(azimuths)
    pos_azimuths = np.array(azimuths)
    
    ax.scatter(pos_azimuths, elevations, color = "black")

    if show_labels:
        assert len(labels) == len(elevations)
        
        for label, azimuth, elevation in zip(labels, pos_azimuths, elevations):
            ax.text(azimuth + 0.5, elevation, label, fontsize = fs, ha = "left", va = "center")
    
    ax.set_xlabel("Azimuth [deg]", fontsize = fs)
    ax.set_ylabel("Elevation [deg]", fontsize = fs)

    cur_xlim = ax.get_xlim()
    ax.set_xlim([cur_xlim[0], cur_xlim[0] + (cur_xlim[1] - cur_xlim[0]) * 1.1])
    
    ax.set_title(plot_title, fontsize = fs)

    ax.text(0.05, 0.95, inner_label, fontsize = fs, transform = ax.transAxes)
    
    ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
    ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)        
    
    fig.savefig(outpath)
    plt.close()

def show_map(ax, intmap, axis_a, axis_b, cmap = "bwr", aspect = "auto", xlim = None, ylim = None, vmin = None, vmax = None, fs = 13, symcolor = True):

    intmap_to_plot = np.squeeze(intmap["map"])
    
    if vmin is None:
        vmin = np.min(intmap_to_plot)

    if vmax is None:
        vmax = np.max(intmap_to_plot)

    if symcolor:
        cscale = max(abs(vmin), abs(vmax))
        vmin, vmax = -cscale, cscale

    im = ax.imshow(np.flip(np.transpose(intmap_to_plot), axis = 0),
                   extent = [intmap[axis_a][0] * defs.cvac, intmap[axis_a][-1] * defs.cvac,
                             intmap[axis_b][0] * defs.cvac, intmap[axis_b][-1] * defs.cvac],
                   cmap = cmap, vmax = vmax, vmin = vmin, aspect = aspect, interpolation = "bicubic")
    
    if xlim:
        ax.set_xlim(*xlim)

    if ylim:
        ax.set_ylim(*ylim)

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

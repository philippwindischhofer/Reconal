import numpy as np
import defs

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
import matplotlib.cm as cmx

def make_xy_plot(outpath, series, labels, colors, xlabel = "", ylabel = "", fs = 13):
        
    fig = plt.figure(figsize = (5, 5), layout = "constrained")
    gs = GridSpec(1, 1, figure = fig)
    ax = fig.add_subplot(gs[0])

    for cur_series, cur_label, cur_color in zip(series, labels, colors):
        xvals, yvals = cur_series
        ax.scatter(xvals, yvals, color = cur_color, label = cur_label)

    ax.set_xlabel(xlabel, fontsize = fs)
    ax.set_ylabel(ylabel, fontsize = fs)

    ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
    ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)        

    ax.legend(frameon = False, fontsize = fs)
    
    fig.savefig(outpath)
    plt.close()

def make_direction_plot(outpath, elevations, azimuths, obs_values, obs_label = "", plot_title = "", labels = [],
                        show_labels = True, fs = 13, cmap = "plasma", inner_label = "", fitline = None, epilog = None):

    fig = plt.figure(figsize = (6, 6), layout = "constrained")
    gs = GridSpec(1, 1, figure = fig)
    ax = fig.add_subplot(gs[0])

    pos_azimuths = np.array(azimuths)
    neg_azimuths = -np.array(azimuths)

    cmap = plt.get_cmap('plasma')
    cmap.set_bad(color = 'gray')
    cNorm  = colors.Normalize(np.nanmin(obs_values), np.nanmax(obs_values))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    colorVals = scalarMap.to_rgba(obs_values)
    
    ax.scatter(neg_azimuths, elevations, color = colorVals)

    cbar = fig.colorbar(scalarMap, ax = ax)
    cbar.set_label(obs_label, fontsize = fs)
    cbar.ax.tick_params(labelsize = fs)
    
    if show_labels:
        assert len(labels) == len(elevations)
        
        for label, azimuth, elevation in zip(labels, neg_azimuths, elevations):
            ax.text(azimuth + 1.0, elevation, label, fontsize = fs - 5, ha = "left", va = "center", color = "gray")
    
    ax.set_xlabel("Azimuth [deg]", fontsize = fs)
    ax.set_ylabel("Elevation [deg]", fontsize = fs)

    cur_xlim = ax.get_xlim()
    # ax.set_xlim([cur_xlim[0], cur_xlim[0] + (cur_xlim[1] - cur_xlim[0]) * 1.15])
    ax.set_xlim([-180.0, 180.0])
    
    ax.set_title(plot_title, fontsize = fs)

    ax.text(0.05, 0.95, inner_label, fontsize = fs, transform = ax.transAxes)
    
    ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
    ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)        

    if epilog:
        epilog(ax)
    
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

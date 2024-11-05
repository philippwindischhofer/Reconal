import argparse, pickle, os, defs, utils
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def show_interferometric_reco(recopath, outpath,
                              epilog = None, show_surface = False, show_arrival = False, fs = 13,
                              show_maxcorr_point = False, autocolor = False):

    with open(recopath, 'rb') as recofile:
        reco = pickle.load(recofile)

    intmap = reco["intmap"]

    # Figure out how to plot this map:
    plot_axes = []
    plot_axes_ind = []
    for ind, axis in enumerate(["x", "y", "z"]):
        if len(intmap[axis]) > 1:
            plot_axes.append(axis)
            plot_axes_ind.append(ind)
        else:
            slice_axis = axis
            slice_val = intmap[slice_axis][0]
            
    if len(plot_axes) != 2:
        raise RuntimeError("Error: can only plot 2d maps!")

    axis_a, axis_b = plot_axes    
    intmap_to_plot = np.squeeze(intmap["map"])

    figsize = (4.6, 4)
    if slice_axis == "z":
        aspect = 1.0
    else:
        aspect = 2.0
    
    fig = plt.figure(figsize = figsize, layout = "constrained")
    gs = GridSpec(1, 1, figure = fig)
    ax = fig.add_subplot(gs[0])

    if autocolor:
        vmin = None
        vmax = None
    else:    
        cscale = np.max(np.abs(intmap["map"]))
        vmax = cscale
        vmin = -cscale
        
    im = ax.imshow(np.flip(np.transpose(intmap_to_plot), axis = 0),
                   extent = [intmap[axis_a][0] * defs.cvac, intmap[axis_a][-1] * defs.cvac,
                             intmap[axis_b][0] * defs.cvac, intmap[axis_b][-1] * defs.cvac],
                   cmap = "bwr", vmax = vmax, vmin = vmin, aspect = aspect)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
    cbar.ax.tick_params(labelsize = fs)
    cbar.set_label("Correlation", fontsize = fs)
    
    ax.set_xlabel(f"{axis_a} [m]", fontsize = fs)
    ax.set_ylabel(f"{axis_b} [m]", fontsize = fs)
    ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
    ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)

    event_number = reco["event"]["meta"]["event"]
    run_number = reco["event"]["meta"]["run"]
    ax.set_title(f"Run {run_number}, Event {event_number}", fontsize = fs)

    ax.text(0.05, 0.92, f"{slice_axis} = {slice_val * defs.cvac:.1f} m", transform = ax.transAxes, fontsize = fs)
    
    if show_surface:
        ax.axhline(0.0, ls = "dashed", color = "gray")

    if show_maxcorr_point:
        maxcorr_point = utils.get_maxcorr_point(reco["intmap"])
        ax.scatter(maxcorr_point[plot_axes[0]] * defs.cvac, maxcorr_point[plot_axes[1]] * defs.cvac, color = "white", marker = "*")

    ax.legend(loc = 'best', frameon = False)
        
    if epilog is not None:
        epilog(ax)
    
    fig.savefig(outpath, dpi = 300)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--recos", nargs = "+", action = "store", dest = "recopaths")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--autocolor", action = "store_true", dest = "autocolor", default = False)
    args = parser.parse_args()

    outdir = args.outdir
    recopaths = args.recopaths
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    for recopath in recopaths:
        outpath = os.path.join(outdir,
                               os.path.splitext(os.path.basename(recopath))[0] + ".pdf")
        show_interferometric_reco(recopath, outpath, autocolor = args.autocolor)

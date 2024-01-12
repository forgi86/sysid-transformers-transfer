import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

LINEWIDTH_IFAC_CONF = 251.8068
LINEWIDTH_L_CSS = 251.8068

tex_fonts = {

    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


# matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 14})
# #matplotlib.rc('text', usetex=True)
# matplotlib.rc('axes', grid=True)

plt.rcParams.update(tex_fonts) # use latex fonts
plt.rcParams.update({"axes.grid": True})

fig, ax = plt.subplots(1, 1, figsize=(3.4842507264425073, 2.1533853742679816))

out = []
errs_trained = []
errs_initial = []
for s in range(50):
    arr = np.load(f"models/outputs_model_wh_seed_{s+1}.npy")
    out.append(arr[0])
    errs_initial.append(arr[0] - arr[1])
    errs_trained.append(arr[0] - arr[2])

#ax.plot(np.array(out).T, c='black')

# Calculate the quantiles
quantile_25 = np.quantile(np.abs(np.array(errs_initial)), q=0.25, axis=0)
quantile_75 = np.quantile(np.abs(np.array(errs_initial)), q=0.75, axis=0)

quantile_25_t = np.quantile(np.abs(np.array(errs_trained)), q=0.25, axis=0)
quantile_75_t = np.quantile(np.abs(np.array(errs_trained)), q=0.75, axis=0)

# Fill between the quantiles
ax.fill_between(np.arange(len(quantile_25)), quantile_25, quantile_75, color='red', alpha=0.3, label='Pre-trained')
ax.fill_between(np.arange(len(quantile_25_t)), quantile_25_t, quantile_75_t, color='green', alpha=0.3, label='Fine-tuned')

# ax.plot(np.median(np.abs(np.array(errs_initial)), axis=0), c='red', label='1')
# ax.plot(np.median(np.abs(np.array(errs_trained)), axis=0), c='green', label='2')

plt.legend()
plt.ylabel("Absolute error")
plt.xlabel("Time step (-)")
plt.savefig("wh_error_25_75_percentile.pdf", format="pdf", bbox_inches="tight")
plt.show()

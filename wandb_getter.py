import wandb
import matplotlib.pyplot as plt
import numpy as np

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

fig, ax = plt.subplots(1, 1, figsize=(3.4842507264425073 * 3 / 4, 2.1533853742679816))

api = wandb.Api()

all_losses = []
all_losses_val = []
x = [i * 100 for i in range(100)]

for s in range(50):
    runs = api.runs("leon-pura/sysid-transfer", filters={"display_name": f"test pwh seed:{s+1}"})

    run = runs[0]

    history = run.scan_history()
    losses = [row["loss"] for row in history]
    losses_val = [row["loss_val"] for row in history]

    ax.plot(x, losses_val)

    all_losses.append(losses)
    all_losses_val.append(losses_val)

all_losses = np.array(all_losses)
all_losses_val = np.array(all_losses_val)
mean_loss = np.mean(all_losses_val, axis=0)
ax.plot(x, mean_loss, c='black', linewidth=2, label='Mean loss')

# runs = api.runs("leon-pura/sysid-transfer", filters={"display_name": "infinite data"})
# run = runs[0]
#
# history = run.scan_history()
# losses_inf = [float(row["loss"]) for row in history]
# losses_inf = np.array(losses_inf)[2:102]
# plt.plot(losses_inf, c='red', linewidth=4)

plt.xlabel("Iteration")
plt.ylabel("Validation loss")

# plt.title("WH re-tuning, training loss")
plt.legend()
plt.savefig("pwh_val_loss.pdf", format="pdf", bbox_inches="tight")
plt.show()

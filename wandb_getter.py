import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()

all_losses = []
for s in range(50):
    runs = api.runs("leon-pura/sysid-transfer", filters={"display_name": f"seed:{s+1}"})

    run = runs[0]

    history = run.scan_history()
    losses = [row["loss_val"] for row in history]

    all_losses.append(losses)

    plt.plot(losses)

all_losses = np.array(all_losses)
mean_loss = np.mean(all_losses, axis=0)
plt.plot(mean_loss, c='black', linewidth=4)

runs = api.runs("leon-pura/sysid-transfer", filters={"display_name": "infinite data"})
run = runs[0]

history = run.scan_history()
losses_inf = [float(row["loss_val"]) for row in history]
losses_inf = np.array(losses_inf)[2:102]

plt.plot(losses_inf, c='red', linewidth=4)
plt.show()

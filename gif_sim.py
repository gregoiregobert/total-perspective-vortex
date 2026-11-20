
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fs = 160
window = 4
channels = ["C3", "C4", "Cz", "Pz"]

t = np.arange(0, 12, 1/fs)

signals = [
    0.5*np.sin(2*np.pi*(9+i)*t) + 0.2*np.random.randn(len(t))
    for i in range(4)
]

# Format carré
fig, axes = plt.subplots(4, 1, figsize=(5, 5), sharex=True)
fig.patch.set_facecolor("#080d18")

lines = []

for ax, signal, channel in zip(axes, signals, channels):
    ax.set_facecolor("#080d18")
    line, = ax.plot([], [], color="#00d9ff", lw=1)
    lines.append(line)
    ax.set_ylim(-1.5, 1.5)
    ax.set_ylabel(channel, color="#00d9ff", rotation=0, labelpad=20)
    ax.grid(alpha=0.2)
    ax.tick_params(colors="white", labelsize=7)

axes[-1].set_xlim(0, window)
axes[-1].set_xlabel("Time (s)", color="white")

fig.suptitle("LIVE EEG MONITORING", color="white", fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.96])

def update(frame):
    start = frame * 8
    end = start + window * fs
    x = t[start:end] - t[start]

    for line, signal in zip(lines, signals):
        line.set_data(x, signal[start:end])

    return lines

ani = FuncAnimation(
    fig, update,
    frames=40,
    interval=100,
    blit=True
)

ani.save(
    "eeg_animation.gif",
    writer=PillowWriter(fps=20),
    dpi=100
)

plt.show()

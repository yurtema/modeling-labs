import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

data = np.load("flight_data.npz")
t = time.time()

positions = data["positions"][:]

num_objects = len(positions[0])
num_frames = len(positions)
print(num_frames)
trajectories = [np.array([frame[a] for frame in positions], dtype=float)
                for a in range(num_objects)]

start_frames = list(map(int, data["t_erup"]))
collided = data["collided"]
radii = data["radii"]

# --- границы по всем ненановым координатам
all_coords = np.concatenate([traj[~np.isnan(traj).any(axis=1)] for traj in trajectories])
xlim = (-1000, 1000)
ylim = (-1000, 1000)
zlim = (0, 3000)

# --- фигура и три сабплота
fig = plt.figure(figsize=(12, 4))
axes = [
    fig.add_subplot(1, 3, 1, projection='3d'),
    fig.add_subplot(1, 3, 2, projection='3d'),
    fig.add_subplot(1, 3, 3, projection='3d')
]

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
fig.tight_layout(pad=0)

# разные углы обзора
axes[0].view_init(elev=0, azim=0)  # вид спереди
axes[1].view_init(elev=20, azim=45)  # основной
axes[2].view_init(elev=0, azim=90)  # вид сбоку

for ax in axes:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

# --- линии и точки для каждой оси
lines_all, points_all = [], []
for ax in axes:
    lines, points = [], []
    for i in range(num_objects):
        color = (1, 0.2, 0.2, 0.8) if collided[i] else (0, 0.3, 0.8, 0.6)
        lw = 2 if collided[i] else 1
        line, = ax.plot([], [], [], lw=lw, color=color)
        point, = ax.plot([], [], [], 'o',
                         markerfacecolor=color,
                         markeredgecolor=color,
                         markersize=radii[i] * 5)
        lines.append(line)
        points.append(point)
    lines_all.append(lines)
    points_all.append(points)


def init():
    for lines, points in zip(lines_all, points_all):
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
    return sum(lines_all + points_all, [])


def update(frame):
    for ax_i in range(3):
        lines, points = lines_all[ax_i], points_all[ax_i]
        for i, traj in enumerate(trajectories):
            start = start_frames[i]
            if frame >= start:
                seg = traj[start:frame + 1]
                valid = ~np.isnan(seg).any(axis=1)
                if np.any(valid):
                    lines[i].set_data(seg[valid, 0], seg[valid, 1])
                    lines[i].set_3d_properties(seg[valid, 2])

                    last_idx = np.where(valid)[0][-1]
                    x, y, z = seg[last_idx]
                    points[i].set_data([x], [y])
                    points[i].set_3d_properties([z])
    return sum(lines_all + points_all, [])


ani = animation.FuncAnimation(
    fig, update, frames=num_frames, init_func=init,
    interval=30, blit=True
)

ani.save('trajectory_views.gif', writer='pillow')
plt.close()
print(time.time() - t)

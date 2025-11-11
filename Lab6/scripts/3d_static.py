import numpy as np
from matplotlib import pyplot as plt

data = np.load("flight_data.npz")

print(list(data))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for a in range(len(data["positions"][0])):
    coords = np.array([i[a] for i in data["positions"]])
    x, y, z = coords.T

    if data["collided"][a]:
        color = 'red'
        linewidth = 2.0
        alpha = 1.0
    else:
        color = 'deepskyblue'
        linewidth = 0.5
        alpha = 0.4

    ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha)

plt.show()

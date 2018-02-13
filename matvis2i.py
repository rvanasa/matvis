import numpy as np
from scipy import linalg as lin

from matplotlib import pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3


def project_3d(arr):
    return [*arr.real, arr.imag[0]]


matrix = np.matrix([
    [0, -1j],
    [1, 0],
])

print(lin.det(matrix))

line_res = 20
anim_speed = 10
bounds = 1

res_steps = 2 * np.pi / line_res
data = np.array([[np.cos(t) * np.sin(u) + np.cos(u) * 1j, np.sin(t) * np.sin(u) + np.cos(u) * 1j]
                 for t in np.arange(line_res) * res_steps
                 for u in np.arange(line_res) * res_steps]).T

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlabel('X')
ax.set_xlim3d([-bounds, bounds])
ax.set_ylabel('Y')
ax.set_ylim3d([-bounds, bounds])
ax.set_zlabel('i')
ax.set_zlim3d([-bounds, bounds])

ax.plot(*np.array([[np.cos(t), np.sin(t)] for t in np.arange(line_res + 1) * res_steps]).T, c='lightgrey')

points = ax.scatter(*project_3d(data))

for v, vec in zip(*lin.eig(matrix)):
    print(v, vec)
    ax.plot(*np.array([[0, 0, 0], project_3d(vec)]).T)

inc_matrix = lin.fractional_matrix_power(matrix, anim_speed / 1000)


def display(t):
    global data
    data = np.matmul(inc_matrix, data)
    points._offsets3d = project_3d(data)
    return [points]


ani = animation.FuncAnimation(fig, display, interval=30)
try:
    plt.show()
except:
    pass

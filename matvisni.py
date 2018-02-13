import numpy as np
from scipy import linalg as lin

from matplotlib import pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3

from cirquit import *


def project_3d(a):
    return a[0:3]


matrix = np.matrix([
    [0, 0, 0, 1],
    [0, 0, 1j, 0],
    [-1j, 0, 0, 0],
    [0, 0, 1, 0],
])

print(lin.det(matrix))

num_points = 100
anim_speed = 10
bounds = 1

data = np.random.randn(matrix.shape[0], num_points) / 2

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlabel('X')
ax.set_xlim3d([-bounds, bounds])
ax.set_ylabel('Y')
ax.set_ylim3d([-bounds, bounds])
ax.set_zlabel('Z')
ax.set_zlim3d([-bounds, bounds])

points_real = ax.scatter(*project_3d(data.real))
points_imag = ax.scatter(*project_3d(data.imag), s=5)

for v, vec in zip(*lin.eig(matrix)):
    print(v, vec)
    ax.plot(*np.array([[0, 0, 0], project_3d(vec.real)]).T)

inc_matrix = lin.fractional_matrix_power(matrix, anim_speed / 1000)


def display(t):
    global data
    data = np.matmul(inc_matrix, data)
    real = project_3d(data.real)
    imag = project_3d(data.imag)
    points_real._offsets3d = real
    points_imag._offsets3d = real + imag
    return [points_real, points_imag]


ani = animation.FuncAnimation(fig, display, interval=30)
try:
    plt.show()
except Exception:
    pass

import numpy as np
from scipy import linalg as lin

from matplotlib import pyplot as plt
from matplotlib import animation

from cirquit import *

matrix = np.matrix([
    [0, 1],
    [1, 0],
])

line_res = 100
anim_speed = 5

data = np.matrix([[np.cos(t ** 2), np.sin(t ** 2)] for t in np.arange(line_res + 1) * 2 * np.pi / line_res]).T

fig = plt.figure()
plt.axis('equal')
real, imag = plt.plot(*data.real, *data.imag.T)[0:2]


def display(t):
    move_data = lin.fractional_matrix_power(matrix, t * anim_speed / 1000) * data
    real.set_data(*move_data.real)
    imag.set_data(*move_data.imag)
    return [real, imag]


ani = animation.FuncAnimation(fig, display, interval=20, blit=True)
plt.show()

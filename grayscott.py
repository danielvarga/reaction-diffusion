
# Gray-Scott (Turing) cellular automaton in Theano.

# Loosely based on Theano heat equation code from
# https://gist.github.com/wiseodd/c08d5a2b02b1957a16f886ab7044032d

# Gray-Scott formula originally found at
# https://github.com/pmneila/jsexp
# and https://mrob.com/pub/comp/xmorphia/


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import theano as th
from theano import tensor as T


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

def draw_plot(x, y, U_orig):
    U = U_orig[:, :, [0,2,1]]
    ax.clear()
    ax.axis("off")
    ax.imshow(U)
    plt.pause(1e-6)
    plt.show()


m = 200
mesh_range = np.arange(-1, 1, 2./(m-1))
x_arr, y_arr = np.meshgrid(mesh_range, mesh_range)

# Initialize variables



U1_arr = np.exp(-5 * ((x_arr-0.5)**2 + (y_arr-0.5)**2)).astype(np.float32)
U1_arr = np.random.uniform(size=U1_arr.shape).astype(np.float32)
U2_arr = np.exp(-5 * ((x_arr+0.5)**2 + (y_arr-0.5)**2)).astype(np.float32)
U2_arr = np.random.uniform(size=U1_arr.shape).astype(np.float32)
U3_arr = np.zeros_like(U1_arr)
U_arr = np.stack((U1_arr, U2_arr, U3_arr), axis=-1)

for i in range(2):
    U_arr[:, :, i] = 0
    for k in range(100):
        px, py = np.random.uniform(low=-1, high=+1, size=2)
        U_arr[:, :, i] += np.exp(-100 * ((x_arr-px)**2 + (y_arr-py)**2)).astype(np.float32)
    U_arr[:, :, i] += np.min(U_arr[:, :, i])
    U_arr[:, :, i] /= np.max(U_arr[:, :, i])


U = T.tensor3("U")

draw_plot(x_arr, y_arr, U_arr)

n = list(range(1, m-1)) + [m-2]
e = n
s = [0] + list(range(0, m-2))
w = s


feed = 0.037
kill = 0.06
delta = 1.0

def grayscott_step(U):
    lapl = U[n, :]+U[:, e]+U[s, :]+U[:, w] - 4*U
    r = U[:, :, 0]
    g = U[:, :, 1]
    du = 0.2097*lapl[:, :, 0] - r*g*g + feed*(1.0 - r)
    dv = 0.105 *lapl[:, :, 1] + r*g*g - (feed+kill)*g
    dst = T.stack((r+du, g+dv, U[:, :, 2]), axis=-1)
    return dst


k = 10

# Batch process the PDE calculation, calculate together k steps
result, updates = th.scan(fn=grayscott_step, outputs_info=U, n_steps=k)
final_result = result[-1]
calc_grayscott = th.function(inputs=[U], outputs=final_result, updates=updates)

U_step = U_arr

for it in range(20000):
    # Every k steps, draw the graphics
    U_step = calc_grayscott(U_step)
    draw_plot(x_arr, y_arr, U_step)

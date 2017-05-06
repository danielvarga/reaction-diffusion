import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import time


INTERACTIVE = True


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

def draw_plot(x, y, U):
    if INTERACTIVE:
        ax.clear()
        ax.axis("off")
        ax.imshow(U)
        plt.pause(1e-6)
        plt.show()


m = 200
mesh_range = np.arange(-1, 1, 2./(m-1))
x_arr, y_arr = np.meshgrid(mesh_range, mesh_range)

# Initialize variables

K = 20

U_arr = np.random.randint(low=0, high=K, size=(m, m))

def demon_step(U):
    r = 1
    Uprime = U.copy()
    for i in range(m):
        for j in range(m):
            nxt = (U[i, j] + 1) % K
            q = (U[max((i-r,0)):i+r+1, max((j-r,0)):j+r+1]==nxt).sum()
            if q>0:
                Uprime[i, j] = nxt
    return Uprime

U_step = U_arr

for it in range(20000):
    # print "starting batch", it
    U_step = demon_step(U_step)
    if INTERACTIVE:
        draw_plot(x_arr, y_arr, U_step)

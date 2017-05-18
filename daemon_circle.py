import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import time


INTERACTIVE = True


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

def draw_plot(x, y, U, it):
    if INTERACTIVE:
        ax.clear()
        ax.axis("off")
        ax.imshow(U)
        plt.pause(1e-6)
        plt.show()
    else:
        ax.clear()
        ax.axis("off")
        ax.imshow(U)
        plt.savefig("frames/%04d.png" % it)

m = 200
mesh_range = np.arange(-1, 1, 2./(m-1))
x_arr, y_arr = np.meshgrid(mesh_range, mesh_range)

# Initialize variables

K = 110

U_arr = np.random.randint(low=0, high=K, size=(m, m))

# U_arr *= 0

l = m/2-m/5
h = m/2+m/5
if False:
    for i in range(l, h):
        U_arr[l, i] = (l+i) % K
        U_arr[i, h] = (h+i) % K
        # U_arr[h, i] = (h+i) % K
        # U_arr[i, l] = (l+i) % K

taken = set()
col = 0
for d in np.arange(0.0, 2*np.pi, 2*np.pi/10000):
    x = int(m/2 + np.cos(d) * m/4)
    y = int(m/2 + np.sin(d) * m/4)
    if (x,y) in taken:
        continue
    U_arr[x, y] = col
    taken.add((x,y))
    col = (col+1)%K

RADIUS = 1

def demon_step(U, mask):
    Uprime = U.copy()
    for i in range(m):
        for j in range(m):
            nxt = (U[i, j] + 1) % K
            box = U[max((i-RADIUS,0)):i+RADIUS+1, max((j-RADIUS,0)):j+RADIUS+1]
            q = (box==nxt).sum()
            if q>0:
                Uprime[i, j] = nxt
                mask[i, j] = 1
    return Uprime, mask

U_step = U_arr
mask = np.zeros_like(U_arr)

for it in range(20000):
    print "starting batch", it
    if it<20:
        mask *= 0
    U_step, mask = demon_step(U_step, mask)
    draw_plot(x_arr, y_arr, U_step * mask, it)

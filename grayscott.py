
# Gray-Scott (Turing) cellular automaton in Theano.

# Loosely based on Theano heat equation code from
# https://gist.github.com/wiseodd/c08d5a2b02b1957a16f886ab7044032d

# Gray-Scott formula originally found at
# https://github.com/pmneila/jsexp
# and https://mrob.com/pub/comp/xmorphia/


import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import time

import theano as th
from theano import tensor as T


INTERACTIVE = True


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

def draw_plot(x, y, U_orig):
    U = U_orig[:, :, [0,2,1]]
    U[:, :, 0] = 1 - U[:, :, 0]
    scipy.misc.imsave("vis.png", U)

    if INTERACTIVE:
        ax.clear()
        ax.axis("off")
        ax.imshow(U)
        plt.pause(1e-6)
        plt.show()


m = 200
mesh_range = np.arange(-1, 1, 2./m)
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

n = list(range(1, m)) + [m-1]
e = n
s = [0] + list(range(0, m-1))
w = s

Du = 0.2297 # 0.2097 in https://github.com/pmneila/jsexp/blob/master/grayscott/index.html#L55
Dv = 0.105
preset = "default"
if preset=="worms":
    feed, kill = 0.078, 0.061
elif preset=="default":
    feed, kill = 0.037, 0.060

delta = 1.0

def grayscott_step(U):
    lapl = U[n, :]+U[:, e]+U[s, :]+U[:, w] - 4*U
    u = U[:, :, 0]
    v = U[:, :, 1]
    du = Du*lapl[:, :, 0] - u*v*v + feed*(1.0 - u)
    dv = Dv*lapl[:, :, 1] + u*v*v - (feed+kill)*v
    dst = T.stack((u+delta*du, v+delta*dv, U[:, :, 2]), axis=-1)
    return dst


diamond = np.array([[0,1,0], [1,0,1], [0,1,0]]).reshape((1,1,3,3)).astype(np.float32)
filters = th.shared(diamond)


# Currently not working, and it's not faster than the smart-indexing solution anyway.
def grayscott_step_convolutional(U):
    u = U[:, :, 0]
    v = U[:, :, 1]
    def conv(x):
        return T.nnet.conv2d(x.reshape((1, 1, m, m)), filters,
                input_shape=(1, 1, m, m), filter_shape=(1, 1, 3, 3), border_mode='half')[0, 0]
    du = Du * conv(u) - u*v*v + feed*(1.0 - u)
    dv = Dv * conv(v) + u*v*v - (feed+kill)*v
    dst = T.stack((u+delta*du, v+delta*dv, u), axis=-1)
    return dst


k = 100

# Batch process k automaton steps together:
result, updates = th.scan(fn=grayscott_step, outputs_info=U, n_steps=k)
assert len(updates)==0
final_result = result[-1]

TEST_GRAD = False
if TEST_GRAD:
    print "building grad"
    dOdU = T.grad(final_result[10, 10, 1], U)
    print "compiling grad"
    f = th.function(inputs=[U], outputs=dOdU)
    print "calculating grad"
    grad = f(U_arr)[:, :, 0]
    print "maximal gradient:", np.abs(grad).max()
    print "done"

calc_grayscott = th.function(inputs=[U], outputs=final_result)

U_step = U_arr

print "batch size", k

for it in range(2000 // k):
    print "starting batch", it
    # U_step += np.random.normal(scale=0.06, size=U_step.shape)
    U_step = calc_grayscott(U_step)
    U_step[:, :, 2] = 0
    if INTERACTIVE:
        draw_plot(x_arr, y_arr, U_step)

print "ended"

draw_plot(x_arr, y_arr, U_step)

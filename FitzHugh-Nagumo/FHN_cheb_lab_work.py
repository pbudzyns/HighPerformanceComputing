import numpy as np
from numpy.fft import fft, ifft

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def chebfft_2d(vv):

    uxx = np.zeros((N + 1, N + 1))
    uyy = np.zeros((N + 1, N + 1))
    ii = np.arange(1, N)

    for i in range(1, N):  # 2nd order derivative using Chebyshev space and FFT
        v = vv[i, :]
        V = list(v) + list(np.flipud(v[ii]))
        U = np.real(fft(V))
        w1_hat = 1j * np.zeros(2 * N)
        w1_hat[0:N] = 1j * np.arange(0, N)
        w1_hat[N + 1:] = 1j * np.arange(-N + 1, 0)
        W1 = np.real(ifft(w1_hat * U))
        w2_hat = 1j * np.zeros(2 * N)
        w2_hat[0:N + 1] = np.arange(0, N + 1)
        w2_hat[N + 1:] = np.arange(-N + 1, 0)
        W2 = np.real(ifft((-w2_hat ** 2) * U))
        uxx[i, ii] = W2[ii] / (1 - x[ii] ** 2) - (x[ii] * W1[ii]) / (1 - x[ii] ** 2) ** (3.0 / 2)  # note that uxx (and uyy) remain 0 for the boundaries
    for j in range(1, N):  # same for Y axis
        v = vv[:, j]
        V = list(v) + list(np.flipud(v[ii]))
        U = np.real(fft(V))
        w1_hat = 1j * np.zeros(2 * N)
        w1_hat[0:N] = 1j * np.arange(0, N)
        w1_hat[N + 1:] = 1j * np.arange(-N + 1, 0)
        W1 = np.real(ifft(w1_hat * U))
        w2_hat = 1j * np.zeros(2 * N)
        w2_hat[0:N + 1] = np.arange(0, N + 1)
        w2_hat[N + 1:] = np.arange(-N + 1, 0)
        W2 = np.real(ifft(-(w2_hat ** 2) * U))
        uyy[ii, j] = W2[ii] / (1 - y[ii] ** 2) - y[ii] * W1[ii] / (1 - y[ii] ** 2) ** (3.0 / 2.0)

    return uxx, uyy


# Global definition
N = 20
x = np.cos(np.pi*np.arange(0, N+1)/N)
y = x
t = 0.0
dt = 6.0 / (N**2)
# dt = 0.001
max_t = 2.0
# max_iter = int(round(max_t / dt))
max_iter = 40
xv, yv = np.meshgrid(x, y)

# Initial state definition
u0 = 0.5*np.cos(xv*(np.pi/2)) + 0.4*np.sin(yv*(np.pi/2))
v0 = 0.2*np.cos(xv*(np.pi/2)) + 0.9*np.sin(yv*(np.pi/2))
# a0 = np.exp(-40 * ((xv - 0.4) ** 2 + yv ** 2))

# Constants
tau = 1.1
Lambda = 1.2
sigma = 0.4
Du = 0.00028
Dv = 0.005
kappa = -0.05

u = np.zeros(shape=(max_iter, N+1, N+1))
v = np.zeros(shape=(max_iter, N+1, N+1))
# a = np.zeros(shape=(max_iter, N + 1, N + 1))

# a[0] = a0
u[0] = u0
v[0] = v0

# a_old = a0
u_old = u0
v_old = v0

for iter in range(1, max_iter):

    uxx, uyy = chebfft_2d(u0)
    vxx, vyy = chebfft_2d(v0)
    # axx, ayy = chebfft_2d(a[iter - 1])

    u_new = (dt**2 * (uxx+uyy))*Du - kappa + Lambda*u0 - u0**3 - sigma*v0
    v_new = ((dt**2 * (vxx+vyy))*Dv + u0 - v0)*(1/tau)

    u_old = u0
    v_old = v0

    u0 = u_new
    v0 = v_new

    # a_new = (dt**2 * (axx+ayy)) + 2 * a0 - a_old
    # a_old = a0
    # a0 = a_new

    u[iter] = u0
    v[iter] = v0
    # a[iter] = a0


def animate(i):
    ax.clear()
    ax.set_zlim([0, 0.8])
    ax.plot_surface(xv, yv, u[i], cmap='coolwarm', linewidth=0, rstride=2, cstride=2)
    # ax.plot_surface(xv, yv, v[i], cmap='coolwarm', linewidth=0, rstride=2, cstride=2)
    ax.set_title('%03d' % (i))
    return ax


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ani = animation.FuncAnimation(fig, animate, np.arange(1, max_iter), interval=1, blit=False)

plt.show()

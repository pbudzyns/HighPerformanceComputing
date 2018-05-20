#!/usr/local/bin/python3
# solving 2d Turing reaction-diffusion system of equations using pseudospectral methods
# The equations are a simplified version of FitzHugh-Nagumo system which describes firing of a neuron
# dT/dt = d²T/dx² + d²T/dy² + r · T · (1 - T) = d²T/dx² + d²T/dy² + R (T)
# assuming diffusion coefficient = 1

# suggested parameters:
# τ = 0.1, du2 = 0.00028, dv2 = 0.005, κ = - 0.05

from numpy import *
from numpy.fft import fft2, ifft2, fftfreq

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import time

#some global definitions
N = 50 #512   # number of points
L = 2*pi #10.   # domain size

x = linspace(0.,L,N)
y = linspace(0.,L,N)
xv, yv = meshgrid(x, y)

#Initial temperature distribution
#u0 = 2 + sin(xv) + sin(2*yv)
#u0 = 0.5 + 0.25 * sin(xv) + 0.25 * sin(2*yv)
u0 = 0.5 + 0.05 * sin(2 * xv + 1) + 0.05 * sin(2  * yv + 1.5)

v0 = 0.5 + 0.01 * sin(25 * xv) + 0.01 * sin(25  * yv + 0.5)

kappa = -1
sigma = 5
d_coeff = 0.1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(xv, yv, u0)
#plt.show()

dt = 0.001  #0.01    # temporal step; increase it a bit and you'll get numerical instability
max_iter = 10000  # number of iterations in time

#FFT the initial distribution
fourier_u = fft2(u0)
fourier_v = fft2(v0)

step = L/N       # spatial step
freq = L*fftfreq(N, d=step)
xfreqs = tile(freq, (N, 1))
yfreqs = xfreqs.transpose()
#freq[0] = 0.01
#print(xfreqs)
#print(yfreqs)

#sum of the X and Y frequencies squared; since we are in 2D now, we use this instead of the squared frequency from the 1D example.
freqs2sum = xfreqs*xfreqs + yfreqs*yfreqs
# store results here
u = zeros(shape=(max_iter,N,N))
v = zeros(shape=(max_iter,N,N))

u[0] = u0
v[0] = v0

#explicit Euler timestepping
for iter in range(1, max_iter) :
    # multiplication is faster than convolution, and we have the IFFT for the previous step anyway; so we just calculate the term in real space instead of calculating a convolution in Fourier space 
    R = u[iter-1] * (1 - u[iter-1] * u[iter-1] / 3) - kappa - sigma * v[iter-1]
    R_fs = fft2(R) # inhomogeneous term for u derivative in Fourier space
    
    S = 10 * (u[iter-1] - v[iter-1])
    S_fs = fft2(S) # inhomogeneous term for v derivative in Fourier space
        
    fourier_u = fourier_u + dt*(-freqs2sum*fourier_u*d_coeff + R_fs);     # next time step in Fourier space
    fourier_v = fourier_v + dt*(-freqs2sum*fourier_v*d_coeff + S_fs);     # next time step in Fourier space

    u[iter] = real_if_close(ifft2(fourier_u)); # IFFT to physical space
    v[iter] = real_if_close(ifft2(fourier_v)); # IFFT to physical space

#fig, ax = plt.subplots()

#ax.plot_wireframe(xv, yv, u[1])
#ax.plot_wireframe(xv, yv, u[999])
#plt.show()

def animate(i):
    ax.clear()
    ax.set_zlim([0,1.1])
    # ax.plot_surface(xv, yv, u[i], cmap='coolwarm', linewidth=0, rstride=2, cstride=2)
    ax.plot_surface(xv, yv, v[i], cmap='coolwarm', linewidth=0, rstride=2, cstride=2)
    
    ax.set_title('%03d'%(i)) 
    return ax
    
ani = animation.FuncAnimation(fig, animate, arange(1, max_iter, 10), interval=1, blit=False, repeat_delay=500)

plt.show()
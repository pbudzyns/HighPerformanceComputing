#!/usr/local/bin/python3
#solving 1d reaction-diffusion equation using FFT
# dT/dt = d²T/dx² + r · T · (1 - T) = d²T/dx² + R (T)
# assuming heat conductivity = 1

from numpy import *
from numpy.fft import fft, ifft, fftfreq

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

#some global definitions
N = 50 #512   # number of points
L = 2*pi #10.   # domain size

x = linspace(0.,L,N)
#Initial component distribution
#u0 = 2 + sin(x) + sin(2*x)
u0 = 0.5 + 0.25 * sin(x) + 0.25 * sin(2*x)
#u0 = 0.9 + 0 * x
r = -2.5

dt = 0.001  #0.01    # temporal step; increase it a bit and you'll get numerical instability
max_iter = 1000  # number of iterations in time

#FFT the initial distribution
fourier = fft(u0)
step = L/N       # spatial step
freq = L*fftfreq(N, d=step)
#freq[0] = 0.01
#print(freq)

# store results here
u = zeros(shape=(max_iter,N))

u[0] = u0

#explicit Euler timestepping
for iter in range(1, max_iter) :
    # multiplication is faster than convolution, and we have the IFFT for the previous step anyway; so we just calculate the term in real space instead of calculating a convolution in Fourier space 
    R = r * u[iter-1] * (1 - u[iter-1])
    R_fs = fft(R) # inhomogeneous term in Fourier space
    
    fourier = fourier + dt*(-freq*freq*fourier + R_fs);     # next time step in Fourier space
    u[iter] = real_if_close(ifft(fourier)); # IFFT to physical space 

fig, ax = plt.subplots()

line, = ax.plot(x, u0)

def animate(i):
    line.set_ydata(u[i])
    return line,
    
ani = animation.FuncAnimation(fig, animate, arange(1, max_iter), interval=50, blit=True, repeat_delay=500)

plt.show()
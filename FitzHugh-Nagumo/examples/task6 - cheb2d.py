#!/usr/local/bin/python3
# solving 2d wave equation using pseudospectral Chebyshev methods

from numpy import *
from numpy.fft import fft, ifft, fft2, ifft2, fftfreq

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import time

from matplotlib.pyplot import subplot, figure ,title,axis
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure,subplot,plot,title,axis,xlabel,ylabel
from matplotlib import cm
from scipy.interpolate import interp2d

# source: chebpy @ github
# example of 1st order differentiation in Chebyshev spectral space
def chebfft(v):
    '''Chebyshev differentiation via fft.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book, pp. 78-79
    '''
    N = len(v)-1
    if N == 0:
        w = 0.0 # only when N is even!
        return w
    x  = cos(pi*arange(0,N+1)/N)    # express Chebyshev points on [-1, 1] through equidistant points on [0..pi] 
    ii = arange(0,N)
    V = flipud(v[1:N]); V = list(v) + list(V);  # values for the 2nd half of the circle
    U = real(fft(V))                            
    b = list(ii); b.append(0); b = b + list(arange(1-N,0)); # fftfreq array
    w_hat = 1j*array(b) # expression for 1st order derivative for Fourier space
    w_hat = w_hat * U   # applying the derivative
    W = real(ifft(w_hat)) # transform the derivative back onto the circle in real space
    w = zeros(N+1)
    w[1:N] = -W[1:N]/sqrt(1-x[1:N]**2) # derivative on (-1, 1) from derivative on (0..pi) 
    w[0] = sum(ii**2*U[ii])/N + 0.5*N*U[N] # special expressions for the edges
    w[N] = sum((-1)**(ii+1)*ii**2*U[ii])/N + \
              0.5*(-1)**(N+1)*N*U[N]
    return w


#some global definitions
N = 24
x = cos(pi*arange(0,N+1)/N)
y = x
t = 0.0; dt = (6.0)/(N**2)
max_t = 2.0
max_iter = int (round( max_t / dt))
xv, yv = meshgrid(x,y)
vv = exp(-40*((xv-0.4)**2 + yv**2));
vvold = vv; 

data = zeros(shape=(max_iter,N+1,N+1))
data[0] = vv

#Time stepping Leapfrog Formula:
fig = figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

for n in range(0, max_iter):
    
    uxx = zeros((N+1,N+1)); uyy = zeros((N+1,N+1));
    ii = arange(1,N);
    
    for i in range(1,N):    #2nd order derivative using Chebyshev space and FFT
        v = vv[i,:];          
        V = list(v) + list(flipud(v[ii]));
        U = real(fft(V));
        w1_hat = 1j*zeros(2*N);
        w1_hat[0:N] = 1j*arange(0,N)
        w1_hat[N+1:] = 1j*arange(-N+1,0)
        W1 = real(ifft(w1_hat * U))
        w2_hat = 1j*zeros(2*N);
        w2_hat[0:N+1] = arange(0,N+1)
        w2_hat[N+1:] = arange(-N+1,0)
        W2 = real(ifft((-w2_hat**2) * U))
        uxx[i,ii] = W2[ii]/(1-x[ii]**2) - (x[ii]*W1[ii])/(1-x[ii]**2)**(3.0/2); #note that uxx (and uyy) remain 0 for the boundaries
    for j in range(1,N):    # same for Y axis
        v = vv[:,j]; 
        V = list(v) + list(flipud(v[ii]));
        U = real(fft(V))
        w1_hat = 1j*zeros(2*N);
        w1_hat[0:N] = 1j*arange(0,N)
        w1_hat[N+1:] = 1j*arange(-N+1,0)
        W1 = real(ifft(w1_hat * U))
        w2_hat = 1j*zeros(2*N);
        w2_hat[0:N+1] = arange(0,N+1)
        w2_hat[N+1:] = arange(-N+1,0)
        W2 = real(ifft(-(w2_hat**2) * U))
        uyy[ii,j] = W2[ii]/(1-y[ii]**2) - y[ii]*W1[ii]/(1-y[ii]**2)**(3.0/2.0);
    vvnew = 2*vv - vvold + dt**2 *(uxx+uyy) #new value 
    vvold = vv ;
    vv = vvnew;
    data[n] = vv
    
def animate(i):
    ax.clear()
    ax.set_zlim([0,1.1])
    ax.plot_surface(xv, yv, data[i], cmap='coolwarm', linewidth=0, rstride=2, cstride=2)
    # ax.plot_surface(xv, yv, v[i], cmap='coolwarm', linewidth=0, rstride=2, cstride=2)
    
    ax.set_title('%03d'%(i)) 
    return ax
    
ani = animation.FuncAnimation(fig, animate, arange(1, max_iter, 1), interval=1, blit=False, repeat_delay=500)
# you can use interp2d to plot prettier plots

plt.show()
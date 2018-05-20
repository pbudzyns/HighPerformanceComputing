"""
for iter in range(1, max_iter):
    R_u = -kappa + Lambda*u[iter-1] - u[iter-1]**3 - sigma*v[iter-1]
    R_v = u[iter-1] - v[iter-1]

    R_u_fs = fft2(R_u)
    R_v_fs = fft2(R_v)

    fourier_u = fourier_u + dt*(-freq2sum*fourier_u*Du + R_u_fs)
    fourier_v = fourier_v + dt*(-freq2sum*fourier_v*Dv + R_v_fs)

    u[iter] = np.real_if_close(ifft2(fourier_u))
    v[iter] = np.real_if_close(ifft2(fourier_v))
"""
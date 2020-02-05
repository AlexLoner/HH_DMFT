# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import numba

@numba.njit()
def weight_points(beta, w0, limit):
    '''beta, w0, limit'''
    points = np.zeros(limit)
    i = 0
    while i < limit:
        r, y = beta * np.random.rand(), w0 * np.random.rand()
        if propagator(r, beta, w0) >= y:
            points[i] = r
            i += 1
    return points

@numba.njit()
def alpha_hub(p, spin, s, delta):
    return -1.0 +  0.5 * p + (2 * spin - 1) * s * delta

@numba.njit()
def alpha_ph(p, s, delta):
    return -1.0 +  0.5 * p +  s * delta

@numba.njit()
def alpha(p, spin, s, delta, b):
    if b:
        return alpha_ph(p, s, delta) 
    else:
        return alpha_hub(p, spin, s, delta)

@numba.njit()
def propagator(tau, beta, w0):
    return 0.5 * w0 * (np.exp(-abs(tau) * w0) + np.exp((abs(tau) - beta) * w0)) / (1.0 - np.exp(-beta * w0))

@numba.njit()
def coef_hub(U, beta, k, w0, pu):
    return -4.0 * U * beta / ((k + 1.0) * (4 * pu + (1.0 - pu) * propagator(0, beta, w0)))

@numba.njit()
def coef_ph(y, beta, k, tau, w0, pu):
    return  2 * y * beta / ((k + 1.0) * (4 * pu / propagator(tau, beta, w0) + (1.0 - pu)))

@numba.njit()
def coef(U, y, beta, k, tau, w0, b, pu):
    if b:
        return coef_ph(y, beta, k, tau, w0, pu) 
    else:
        return coef_hub(U, beta, k, w0, pu)        

@numba.njit()
def det(M):
    return np.linalg.det(M)

@numba.njit()
def inv(M):
    return np.linalg.inv(M)

@numba.njit()
def gf(tay, beta, g_tau):
    '''tay, beta, g_tau'''
    if 0 <= tay:
        return g_tau[np.int(np.round((len(g_tau) - 1.0) * tay / beta))] # 9999 = len(g_tau) - 1
    else:
        return -g_tau[np.int(np.round((len(g_tau) - 1.0) * (beta + tay) / beta))]

# @numba.njit()
def comp_gf(mtx, times, t, t1, beta, g_tau):
    '''mtx[i,j] = _M**-1'''
    gf1 = np.array([gf(t - r, beta, g_tau) for r in times])
    gf2 = np.array([gf(s - t1, beta, g_tau) for s in times])
    return gf(t - t1, beta, g_tau) - (gf1.dot(mtx)).dot(gf2)
    
@numba.njit()   
def comp_gf_iw(num, mg, times, freq, g_iw):
    e = np.exp(-1j*freq[num] * times).astype(np.complex64)
    return g_iw[num] * np.complex64(1.0 - e.real.astype(np.float32).dot(mg.astype(np.float32)) - 1j*(e.imag.astype(np.float32).dot(mg.astype(np.float32))))

@numba.njit()
def tau_n(tau, dt, beta):
    sign = 2 * np.random.randint(2) - 1
    tau_n =  tau + dt * sign
    if tau_n > beta:
            tau_n = tau_n - beta
    elif tau_n < 0 :
        tau_n = beta + tau_n
    return tau_n

@numba.njit()
def choice(p):
    return np.int8(np.random.rand() > p)

@numba.njit()
def rewrite(mtx, edge, upto, arr_h, arr_v):
    for i in range(upto):
        mtx[edge, i] = arr_h[i]
    for i in range(upto):
        mtx[i, edge] = arr_v[i]

@numba.njit()
def correlator_computations(GFu, GFd, k, beta, minv0, minv1, g_tau, times0, times1, freq, sign_m, g_iw,num_iw):
    gft0 = np.array([gf(s, beta, g_tau) for s in times0], dtype=np.float32)
    mg0 = minv0.dot(gft0)
    gft1 = np.array([gf(s, beta, g_tau) for s in times1], dtype=np.float32)
    mg1 = minv1.dot(gft1)
    for num in numba.prange(num_iw):
        GFu[k,num] = np.float32(sign_m) * comp_gf_iw(num, mg0, times0, freq, g_iw)
        GFd[k,num] = np.float32(sign_m) * comp_gf_iw(num, mg1, times1, freq, g_iw)

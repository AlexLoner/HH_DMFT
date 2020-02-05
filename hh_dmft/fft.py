# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals

#from pytriqs.gf import *
#from pytriqs.gf.meshes import MeshImTime, MeshImFreq
import numpy as np

def fourier(beta, g_tau, num_iw):
    npoints = len(g_tau)
    num = npoints - 1
    tau = np.arange(0, npoints - 1) * beta * num / (num - 1.0) / float(npoints - 1)
    w_n = np.pi * (2 * np.array(range(-int(num // 2), num // 2 + 1)) + 1) / beta
    gtau = g_tau + 0.5
    giwn = beta * np.roll(np.fft.ifft(gtau[:-1] * np.exp(1j * np.pi * tau / (beta + tau[1])) ),
                          int(num / 2), axis=-1) +  1./ (1.j * w_n)
    center = npoints // 2 - 1
    return giwn[center - num_iw // 2 : center + num_iw // 2]

def inverse_fourier(beta, g_iwn, n_points): 
    num = len(g_iwn)
    w_n = np.pi * (2 * np.array(range(-int(num / 2), num // 2)) + 1) / beta
    giwn = g_iwn - 1./ (1.j * w_n)

    tau = np.arange(0, n_points - 1) * beta * num / (num - 1.0) / float(n_points - 1)
    ft = np.fft.fft(giwn, n_points - 1) * np.exp(1j * np.pi * (num - 2.0) * tau / beta) / beta - 0.5
    ft = np.append(ft, -1 - ft[0])
    return np.real(ft).astype(np.float32)

def w2tau(temp, beta):
    '''Convert green function from matsubara frequencies to imaginary time'''
    temp = np.concatenate((np.conj(temp[::-1]), temp))
    return inverse_fourier(beta, temp, n_points=10000)
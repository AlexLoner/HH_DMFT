#-*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import itertools
import copy
from math import factorial as fact


def bose(m, n):
    '''
    Input
    -----
    m :: int ::: number of sites
    n :: int ::: number of particles in the system
    
    Return
    -----
    Construct basis for bose system :: np.array
    '''
    R = int(fact(n + m - 1)/fact(n)/fact(m - 1))
    b = np.zeros((R,m), dtype=np.int32)
    b[0,m-1] = n
    for i in range(R-1):
        j = m - 1
        while j > 0:
            if b[i,j] in range(2,n+1) :
                b[i+1,:] = b[i,:]
                b[i+1,j] = 0
                b[i+1,j-1] = b[i+1,j-1] + 1
                b[i+1,m-1] = b[i,j] - 1
                break
            elif b[i,j] > 0:
                b[i+1,:] = b[i,:]
                b[i+1,j-1] = b[i+1,j-1] + 1
                b[i+1,j] = b[i,j] - 1
                break
            j -= 1
    return b

def limit_basis(m, n, n_max):
    '''
    Input
    -----
    m     :: int ::: number of sites
    n     :: int ::: number of particles in the system
    n_max :: int ::: maximum number of boson on one site
    Return
    -----
    Construct basis for limited bose system :: np.array
    '''
    R = int(fact(n + m - 1)/fact(n)/fact(m - 1))
    b = bose(m,n)
    f = np.zeros((R,m), dtype=np.int32)
    j = 0
    for i in range(b.shape[0]):
        if any(b[i] > n_max): 
            continue
        else:
            f[j] = b[i]
            j += 1
    return f[:j]

def bose_unsave(m, n):
    '''
    Input
    -----
    m     :: int ::: number of sites
    n     :: int ::: number of particles in the system
    Return
    -----
    Construct basis for grand canononical bose system :: np.array
    '''   
    return np.array( list(map(list, itertools.product(range(n+1),repeat=m))) )

def fermi(m, n_up, n_down):
    '''
    Input
    -----
    m      :: int ::: number of sites
    n_up   :: int ::: number of particles with spin up in the system
    n_down :: int ::: number of particles with spin down in the system
    Return
    -----
    Construct basis for fermi system with respect to the spin:: np.array
    '''     
    R = int((fact(m)/fact(n_up)/fact(m-n_up))*(fact(m)/fact(n_down)/fact(m-n_down)))
    fs = np.zeros((R,2*m), dtype=np.int32)
    part_1 = limit_basis(m,n_up,1)
    if n_up == n_down:
        part_2 = copy.copy(part_1)
    else:
        part_2 = limit_basis(m,n_down,1)
    size_1, size_2 = part_1.shape[0], part_2.shape[0]
    for i in range(size_1):
        for j in range(size_2):
            fs[i*size_2+j] = np.concatenate((part_1[i],part_2[j]), axis=0)
    return fs

def full_basis_save(m_d, m_c, m_b, n_down, n_up, n_max):
    """
    Input
    ----
    m_d    :: int ::: number of site in the cluster
    m_c    :: int ::: number of site in the fermi bath
    m_b    :: int ::: number of site in the boson bath
    n_up   :: int ::: number of particles with spin up in the system
    n_down :: int ::: number of particles with spin down in the system
    n_max  :: int ::: maximum number of boson on one site
    Return
    -----
    Constuct basis (fermi and boson) in the grand conononical ensemble
    """
    mtx_1 = fermi(m_d+m_c, n_up,n_down)
    mtx_2 = bose_unsave(m_b,n_max)
    size_1, size_2 = mtx_1.shape[0], mtx_2.shape[0]
    fb = np.zeros((size_1*size_2,mtx_1.shape[1]+m_b),dtype=np.int32)
    for i in range(size_1):
        for j in range(size_2):
            fb[i*size_2+j] = np.concatenate((mtx_1[i],mtx_2[j]), axis=0)
    return fb

def fermi_spinless(m, n):
    '''
    Input
    -----
    m      :: int ::: number of sites
    n      :: int ::: number of fermions in the system
    Return
    -----
    Construct spinless fermi basis :: np.array
    '''
    return limit_basis(m, n, 1)

def full_basis_save_spinless(md, mc, mb, n, n_max):
    """
    Input
    ----
    md    :: int ::: number of site in the cluster
    mc    :: int ::: number of site in the fermi bath
    mb    :: int ::: number of site in the boson bath
    n     :: int ::: number of fermions in the system
    n_max :: int ::: maximum number of boson on one site
    Return
    -----
    Constuct spinless basis (fermi and boson) in the grand conononical ensemble
    """
    mtx_1 = fermi_spinless(md + mc, n)
    mtx_2 = bose_unsave(mb, n_max)
    size_1, size_2 = mtx_1.shape[0], mtx_2.shape[0]
    fb = np.zeros((size_1 * size_2, mtx_1.shape[1] + mb), dtype=np.int32)
    for i in range(size_1):
        for j in range(size_2):
            fb[i*size_2+j] = np.concatenate((mtx_1[i], mtx_2[j]), axis=0)
    return fb
    
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals

import numba
import torch as th
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

@numba.jit(parallel=True)
def diagonalize_hamiltonian(H, blocks):
    '''...'''
    U_full = csr_matrix(H.shape, dtype=np.float32)
    E_full = np.zeros(H.shape[0])
    for i in numba.prange(len(blocks)):
        block = blocks[i]
        X,Y = np.meshgrid(block,block)
        E, U = np.linalg.eigh(H[X,Y].todense())
        E_full[block] = E
        U_full[Y,X] = U

    E = np.array(E_full)
    E0 = np.min(E)
    E = E - E0
    return E, E0, U_full

def get_frequency_greens_function_component(op2eb, iwn, op1, op2, beta, E, U, Z, xi=-1.0):

    r"""
    Returns:
    G^{(2)}(i\omega_n) = -1/Z < O_1(i\omega_n) O_2(-i\omega_n) >
    """
    op1_eig, op2_eig = op2eb(U, [op1.astype(np.float32), op2.astype(np.float32)])
    # -- Compute Lehman sum for all operator combinations
    G = np.zeros((len(iwn)), dtype=np.complex64)
    op = (op1_eig.getH().multiply(op2_eig)).tocoo().astype(np.float32)
    del op1_eig, op2_eig
    M = (np.exp(-beta * E[op.row]) - xi * np.exp(-beta * E[op.col])) * op.data
    e = (E[op.row] - E[op.col])
    del op
    for i in range(len(iwn)):
        G[i] = np.sum(M / (iwn[i] - e))
    G /= Z
    return G


def calculate_partition_function(beta, E):
    return np.sum(np.exp(-beta * E))

def operators_to_eigenbasis_cpu(U, op_vec):
    dop_vec = []
    u = U.toarray()
    for op in op_vec:
        M = u.T.dot(op.toarray().dot(u))
        dop_vec.append(coo_matrix(M, dtype=np.float32))

    return dop_vec

def operators_to_eigenbasis_gpu(U, op_vec):
    dop_vec = []
    d = th.FloatTensor(U.data)
    rc = th.LongTensor(np.vstack((U.row, U.col)))
    u = th.sparse.FloatTensor(rc, d,  th.Size(U.shape))
    del d, rc
    th.cuda.empty_cache()
    with th.cuda.device(0):
        u = u.cuda()
    for op in op_vec:
        with th.cuda.device(0):
            data = op.data.astype(np.float32)
            d = th.FloatTensor(data)
            rc = th.LongTensor(np.vstack((op.row, op.col)))
            tm = th.sparse.FloatTensor(rc, d,  th.Size(op.shape)).cuda()

            del d, rc
            th.cuda.empty_cache()

            temp = th.spmm(tm, u.to_dense())

            del tm
            th.cuda.empty_cache()

            dop_gpu = th.spmm(u.t(), temp)

            del temp
            th.cuda.empty_cache()

            dop = csr_matrix(dop_gpu.cpu().numpy(), dtype=np.float32)

            del dop_gpu
            th.cuda.empty_cache()
        dop_vec.append(dop)
    del u
    th.cuda.empty_cache()
    return dop_vec

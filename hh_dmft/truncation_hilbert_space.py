# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import torch as th
import numba
import sys
import psutil
from . import anderson_model as h
from . import ed_tools

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import eigs
from datetime import datetime

class THS(object):
    """docstring for ."""
    ELEMENT_SIZE = sys.getsizeof(np.float64(1.0))

    def __init__(self, ham_object, eigvals_number=200):
        self.hamiltonian = ham_object
        self.k = eigvals_number
        self.__diagonalization()
               
    def __diagonalization(self):
        e, U = eigs(self.hamiltonian, k = self.k, which='SR')
        self.U = csr_matrix(U, dtype=np.complex64)
        del U
        self.E = e - min(e)
    
    def __check_ke(self, beta, en):
        kE = np.where(beta * en > self.bE)[0][0]
        self.__flag = True
        return kE
    
    def __trunc(self, beta, bE, operators, cut_C):
        
        self.op = operators
        self.bE = bE
        self.__flag = False
        self.__write_info()
        while self.__flag == False:
            try:
                en = np.array(sorted(self.E))
                kE = self.__check_ke(beta, en)
            except:
                self.k += 50
                self.__diagonalization()
            finally:
                self.__write_info(self.k)
            
        ind_V = np.array([j for i in range(kE) for j in np.where(self.E == en[i])[0][:]])
        
        index = []
        i = 0
        for ind in ind_V:
            V = self.U[:,ind].toarray()
            #for op in all_ops:
            index.append(np.where(abs(V) > cut_C)[0])
            for op in self.op:
                C = op.dot(V)
                index.append(np.where(abs(C) > cut_C)[0])
            i += 1

        cloud = set()
        for i in index:
            cloud = cloud.union(set(i))
        indexes = list(cloud)
        
        # todense <-> toarray
        self.h = self.hamiltonian[indexes, :][:, indexes]
        ram_avlb = psutil.virtual_memory()[1]
        ram_mtx = 2 * THS.ELEMENT_SIZE * self.h.shape[0] ** 2        
        ram_eigenvector_size = self.U.shape[0] * self.U.shape[1] * THS.ELEMENT_SIZE
              
        self.__write_info(ram_avlb, ram_mtx, ram_eigenvector_size)
        
        if ram_avlb > (ram_mtx + ram_eigenvector_size):
            self.h = self.h.toarray()
            self.__success = True
            
        else:
            return self.__success
        
        temp_E, temp_U = np.linalg.eigh(self.h)
        self.E = np.array(temp_E)
        self.U = coo_matrix(temp_U, dtype=np.float32)
        self.E = self.E - np.min(self.E)
        self.Z = ed_tools.calculate_partition_function(beta, self.E)
        self.new_operator = [i.tolil()[indexes, :][:, indexes].tocoo().astype(np.float32) for i in self.op]
        del self.h
        return self.__success
        
    def get_gf_trunc_ratio(self):
        return float(self.hamiltonian.shape[0]) / self.h.shape[0]
    
    def get_GF(self, beta, bE, operators, cut_C, wn):
        
        self.__success = False
        while not self.__success:
            self.__success = self.__trunc(beta, bE, operators, cut_C)
            cut_C *= 1.25
            bE /= 1.25
        try:
            self.GF = ed_tools.get_frequency_greens_function_component(ed_tools.operators_to_eigenbasis_gpu, 1j * wn, self.new_operator[0],  self.new_operator[1], beta, self.E, self.U, self.Z)
            perm = 'gpu'
        except:
            self.GF = ed_tools.get_frequency_greens_function_component(ed_tools.operators_to_eigenbasis_cpu, 1j * wn, self.new_operator[0],  self.new_operator[1], beta, self.E, self.U, self.Z)
            perm = 'cpu'
        finally:
            th.cuda.empty_cache()
        return self.GF
    
    @staticmethod
    def __write_info(*args):
        
        if len(args) < 1:
            _str = str(datetime.now().time()) + '\n'
        elif len(args) == 1:
            _str = 'Eigsvals {}\n'.format(*args)
        else:
            
            #arg = list(map(lambda x : x * 10 ** -9, *args))
            _str = '[bytes] :: Availible {}, Matrix {}, Eigenvectors {}\n'.format(*args)
            v = 10.0 ** -9
            _str += '[Gb]  ::  Availible {}, Matrix {}, Eigenvectors {}\n\n'.format(args[0] * v, args[1] * v, args[2] * v)

        with open('memory_usage.txt', 'a+') as file:
            file.write(_str)

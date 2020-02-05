# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
from datetime import datetime
import numpy as np

class DMFT():
    def __init__(self, n_loops, gf_initial, solver, alpha, tol, tb_model = None):
        self.n_loops = n_loops
        self.solver = solver
        self.alpha = alpha
        self.tol = tol
        self.num_iw = gf_initial.shape[0]
        self.TB = tb_model
        
        ##### Create lists for Greens Function
        self.g0_list = np.zeros((n_loops + 1, self.num_iw), dtype=np.complex64)
        self.g0_list[0, :] = gf_initial
        self.g_list = np.zeros((n_loops, self.num_iw), dtype=np.complex64)
    
    def _check_consistent_condition(self, m1, m2):
        n = self.num_iw // 3
        return np.linalg.norm(m1[self.num_iw//2 :self.num_iw//2 + n] - m2[self.num_iw//2:self.num_iw//2 + n]) < self.tol
    
#     def _gf_calc(self):
        
#         G = GfImFreq(mesh = MeshImFreq(self.solver.beta, 'Fermion', self.num_iw//2), indices = [1])
#         Sigma0 = GfImFreq(mesh = MeshImFreq(self.solver.beta, 'Fermion', self.num_iw//2), indices = [1]); 
#         Sigma0.data[:,0,0] =  self.sigma 
#         G << self.HT(Sigma = Sigma0, mu=self.solver.mu)
#         g = G.data[:,0,0]
#         return g
    
    def body(self):
        self.consistent_condition = False
        cur_loop = 0
        while not self.consistent_condition and cur_loop < self.n_loops:
            print("DMFT Loop : {} / {} starts at {}".format(cur_loop, self.n_loops, datetime.now()))
            self.solver.g0 = self.g0_list[cur_loop, :]
            self.solver.run()
            
            gf_cluster = self.solver.get_gf()                                    # will run the solver to get the cluster (anderson green function)
            self.sigma = self.g0_list[cur_loop, :] ** -1 - gf_cluster ** -1      # sigma = sigma_anderson (cluster)
            self.TB.HT(self.g_list[cur_loop, :], self.solver.wn, self.solver.mu, self.sigma)
#             self.g_list[cur_loop, :] = self._gf_calc()
            self.g0_list[cur_loop + 1, :] = self.alpha * ((self.sigma + self.g_list[cur_loop, :] ** -1) ** -1) + (1.0 - self.alpha) * self.g0_list[cur_loop, :] # should anble to calculate new system's gf with corresponding dispersion
            self.consistent_condition = self._check_consistent_condition(self.g0_list[cur_loop, :], self.g0_list[cur_loop + 1, :])
            cur_loop += 1
        if self.consistent_condition:
            print("DMFT Loop : ends with self-consistent_condition at {} in {} step".format(datetime.now(), cur_loop))
            self.g0_list = np.delete(self.g0_list, range(cur_loop, self.n_loops + 1), axis=0)
            self.g_list = np.delete(self.g_list, range(cur_loop, self.n_loops), axis=0)
        else:
            print("DMFT Loop : {} / {} ends at {}".format(cur_loop, self.n_loops, datetime.now()))

from utils import ed_tools, fit

from hh_dmft.truncation_hilbert_space import THS
from hh_dmft.anderson_model import AModel


class ED:

    def __init__(self, beta, wn, betaEmax, minImpact, full=True, **hamiltonian_params):
        self.ham = AModel(**hamiltonian_params)
        self.wn = wn
        self.beta = beta
        self.g0 = None
        self.bE = betaEmax
        self.cut_C = minImpact
        self.full = full
        self.mu = self.ham.mu


    def run(self):
        self.ham.Vcd, self.ham.ec, d = fit.get_param(
            self.g0, self.beta, self.wn, self.ham.mc, self.ham.mu, self.ham.Vcd, self.ham.ec, (-4.0, 4.0), (-4.0, 4.0)
        )
        self.ham.init_calc()
        if not self.full:
            self.ths = THS(self.ham.get_H(), eigvals_number=15)
        else:
            self.E, self.E0, self.U = ed_tools.diagonalize_hamiltonian(self.ham.get_H(), self.ham.get_blocks())

    def get_gf(self):
        if not self.full:
            return self.ths.get_GF(self.beta, self.bE, [self.ham.c_up(), self.ham.c_dag_up()], self.cut_C, self.wn)
        else:
            Z = ed_tools.calculate_partition_function(self.beta, self.E)
            op1, op2 = self.ham.c_dag_up(), self.ham.c_up()
            try:
                return ed_tools.get_frequency_greens_function_component(
                    ed_tools.operators_to_eigenbasis_gpu, 1j * self.wn, op2, op1, self.beta, self.E, self.U, Z
                )
            except:
                return ed_tools.get_frequency_greens_function_component(
                    ed_tools.operators_to_eigenbasis_cpu, 1j * self.wn, op2, op1, self.beta, self.E, self.U, Z
                )

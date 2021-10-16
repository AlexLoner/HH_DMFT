import tqdm
from numpy.random import rand, randint
from utils.fft import inverse_fourier
from utils.qmc_tools import *


class QMC_CT_INT:

    def __init__(self, **kwargs):
        '''beta, U, num , g0, num_iw, steps, p=1., delta=0.51, y = 0.3, w0 = 0.07, pu=1'''
        self.__dict__.update(kwargs)
        self.n = 1
        self.wn = np.pi * (2 * np.array(range(-int(self.num_iw / 2), self.num_iw // 2)) + 1) / self.beta

    def __reset_calc(self):
        self.conf = {}
        self.times = [[], []]
        self.point = [0, 0]
        self.sign_m = 0.0
        self.sign = 0.0
        self.size = 100 * int(self.beta)
        # self.M = [lil_matrix((self.size, self.size),dtype=np.float64), lil_matrix((self.size, self.size),dtype=np.float64)]
        self.M = [
            np.zeros((self.size, self.size), dtype=np.float64),
            np.zeros((self.size, self.size), dtype=np.float64)
        ]
        self.c = 0  # global var for store config
        self.w_idx = 0  # weight index
        self.g_tau = inverse_fourier(self.beta, self.g0, n_points=10000)
        self.d_tau = weight_points(self.beta, self.w0, limit=int(self.steps * 1.25))
        self._start(self.num)
        self.M_inv = [
            np.linalg.inv(self.M[0][:self.point[0], :self.point[0]]),
            np.linalg.inv(self.M[1][:self.point[1], :self.point[1]])
        ]


    def _start(self, num):
        for _ in range(num):
            b = choice(self.pu)
            site = randint(self.n)
            time = [rand() * self.beta, 0]
            spin = np.random.choice([0, 1], size=2, replace=b)
            s = 2 * choice(0.5) - 1
            if b:
                time[1] = tau_n(time[0], self.d_tau[self.w_idx], self.beta)
            else:
                time[1] = time[0]
            self.conf[self.c] = s, site, time, spin, b
            self.c += 1
            for i, t1 in zip(spin, time):
                self.times[i].append(t1)
                self.M[i] = self._M(s, 1, t1, self.M[i], i, spin, b)
                self.point[i] += 1
            self.w_idx += 1

    def cr_prob(self, spin, time, s, b):
        if spin[0] == spin[1]:
            if self.times[spin[0]]:
                p = - np.eye(2) * alpha(self.p, spin[0], s, self.delta, b)
                for i in range(2):
                    for j in range(2):
                        p[i, j] += comp_gf(self.M_inv[spin[0]], self.times[spin[0]], time[i], time[j], self.beta, self.g_tau)
                return np.linalg.det(p)
            else:
                p = - alpha(self.p, spin[0], s, self.delta, b) * np.eye(2)
                for i in range(2):
                    for j in range(2):
                        p[i, j] += gf(time[i] - time[j], self.beta, self.g_tau)
                return np.linalg.det(p)
        else:
            if self.times[spin[0]]:
                p1 = comp_gf(self.M_inv[spin[0]], self.times[spin[0]], time[0], time[0], self.beta, self.g_tau) - alpha(
                    self.p, spin[0], s, self.delta, b)
            else:
                p1 = gf(0, self.beta, self.g_tau) - alpha(self.p, spin[0], s, self.delta, b)
            if self.times[spin[1]]:
                p2 = comp_gf(self.M_inv[spin[1]], self.times[spin[1]], time[1], time[1], self.beta, self.g_tau) - alpha(
                    self.p, spin[1], s, self.delta, b)
            else:
                p2 = gf(0, self.beta, self.g_tau) - alpha(self.p, spin[1], s, self.delta, b)
            return p1 * p2

    def rm_prob(self, ind, spin, time, s):
        if spin[0] == spin[1]:
            p = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    p[i, j] = self.M_inv[spin[0]][ind[i], ind[j]]
            return np.linalg.det(p)
        else:
            p1 = self.M_inv[spin[0]][ind[0], ind[0]]
            p2 = self.M_inv[spin[1]][ind[1], ind[1]]
            return p1 * p2

    def create(self):
        b = choice(self.pu)
        site = randint(self.n)
        time = [rand() * self.beta, 0]
        spin = np.random.choice([0, 1], size=2, replace=b)
        s = 2 * choice(0.5) - 1
        if b:
            time[1] = tau_n(time[0], self.d_tau[self.w_idx], self.beta)
        else:
            time[1] = time[0]
        k = len(self.conf)
        prob = self.cr_prob(spin, time, s, b)
        self.conf[self.c] = s, site, time, spin, b
        P = coef(self.U, self.y, self.beta, k, time[0] - time[1], self.w0, b, self.pu) * prob
        if rand() < abs(P):
            self.sign_m = np.sign(P)
            for i, t1 in zip(spin, time):
                self.times[i].append(t1)
                self.M[i] = self._M(s, 1, t1, self.M[i], i, spin, b)
                self.point[i] += 1
            self.M_inv[0] = np.linalg.inv(self.M[0][:self.point[0], :self.point[0]])
            self.M_inv[1] = np.linalg.inv(self.M[1][:self.point[1], :self.point[1]])
            self.c += 1
        else:
            del self.conf[self.c]
        self.w_idx += 1

    def remove(self):
        idx = np.random.choice(list(self.conf.keys()))
        s, site, time, spin, b = self.conf[idx]
        ind = [self.times[spin[i]].index(time[i]) for i in [0, 1]]
        k = len(self.conf) - 1
        prob = self.rm_prob(ind, spin, time, s)  # -1 add to mn matrix cause self.point is still pointing
        P = prob / coef(self.U, self.y, self.beta, k, time[0] - time[1], self.w0, b, self.pu)
        if rand() < abs(P):
            self.sign_m = np.sign(P)
            v = 0
            for i, t1 in zip(spin, time):
                self.M[i] = self._M(s, 0, t1, self.M[i], i, spin, b)
                v += 1
                self.point[i] -= 1  # shifting self.point
            self.times[spin[0]][ind[0]] = self.times[spin[0]][-1]
            del self.times[spin[0]][-1]
            ind = self.times[spin[1]].index(time[1])
            self.times[spin[1]][ind] = self.times[spin[1]][-1]
            del self.times[spin[1]][-1]

            del self.conf[idx]
            self.M_inv[0] = np.linalg.inv(self.M[0][:self.point[0], :self.point[0]])
            self.M_inv[1] = np.linalg.inv(self.M[1][:self.point[1], :self.point[1]])

    def _iter(self):
        process = randint(2)
        if len(self.times[0]) <= 2 or len(self.times[1]) <= 2:
            process = 1
        if process:
            self.create()
        else:
            self.remove()

    def mgf(self, l, times):
        '''
        l is a tay_n
        return a new lines for M
        '''
        n = len(times)
        mv = np.zeros(n)
        mh = np.zeros(n)
        for i in range(n):
            t = times[i] - l
            mv[i] = gf(t, self.beta, self.g_tau)
            mh[i] = gf(-t, self.beta, self.g_tau)
        return mh, mv

    def _M(self, s, proc, t, M, idx, spin, b):  # idx = spin
        times = self.times[idx]
        if proc:
            mas_h, mas_v = self.mgf(t, times)
            rewrite(M, self.point[idx], self.point[idx] + 1, mas_h, mas_v)
            M[self.point[idx], self.point[idx]] -= alpha(self.p, idx, s, self.delta, b)
        else:
            i = times.index(t)
            rewrite(M, i, M.shape[0], M[self.point[idx] - 1, :], M[:, self.point[idx] - 1])
        return M

    def get_gf(self):
        return np.mean(0.5 * (self.GFu + self.GFd) / self.sign, axis=0)

    def run(self):

        self.__reset_calc()
        step = 10
        thermosteps = self.steps // 10
        for _ in tqdm.tqdm(range(int(thermosteps)), desc='Thermo: '):
            self._iter()

        self.GFu = np.zeros((self.steps // step, self.num_iw), dtype=np.complex128)
        self.GFd = np.zeros((self.steps // step, self.num_iw), dtype=np.complex128)
        self.occup = np.zeros(self.steps // step, dtype=np.float64)
        k = 0
        for i in tqdm.tqdm(range(self.steps), desc='Simulation: '):
            self._iter()
            if i % step == 0:
                self.occup[k] = self.sign_m * comp_gf(self.M_inv[0], self.times[0], self.beta, 0, self.beta, self.g_tau)
                self.sign = (k * self.sign + self.sign_m) / (1.0 + k)
                correlator_computations(
                    self.GFu, self.GFd, k, self.beta, self.M_inv[0], self.M_inv[1], self.g_tau,
                    np.array(self.times[0]), np.array(self.times[1]), self.wn, self.sign_m, self.g0, self.num_iw
                )
                k = k + 1


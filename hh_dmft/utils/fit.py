# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import numba
import multiprocessing
from scipy.optimize import differential_evolution as de
import warnings
warnings.filterwarnings('ignore')

def get_param(g0, beta, wn, size, mu, V=None, ei=None, bound_v=(-2.0, 2.0), bound_e=(-2.0, 1.5)):
    '''
    Возвращает значение параметров фиттированной функции Грина
    -------
    g0 : Функция Грина :: np.array(compex)
    wn : Матцубаровские частоты :: np.array(float)
    size : Число узлов фермионной ванны :: int
    V, ei : Фитируемые параметры (перескоки с кластера на ванну и затравочная энегрия на ванне) :: 0 or np.array(float)
    bound_v, bound_e : Границы для искомых параметров :: tuple
    \\\\\ Параметры для варьирования хим потенциала
    shift : шаг :: float
    erabs : точность ::  float
    -----
    Returns: V, e, mu :: np.array(float), np.array(float), float
    '''
    # Параметры для минимизации
    if type(V) != int:
        V = V.copy()
        ei = ei.copy()
        V += 0.005 * (2 * np.random.rand(size) - 1) # Добавляем шум к уже найденным параметрам 
        ei += 0.005 * (2 * np.random.rand(size) - 1)
    else:
        V = np.random.rand(size) # Работает при первой итерации
        ei = np.random.rand(size)
#     init = np.concatenate((V, ei))
    bounds = [bound_v] * size + [bound_e] * size
    #num_threads = int(os.popen(u'grep -c cores /proc/cpuinfo').read()) - 1
    num_threads = int(multiprocessing.cpu_count() - 1)
    model = de(diff, bounds=bounds, args=(g0, wn, mu), tol=1e-08, maxiter = 5000, workers=num_threads)
    V = (model.x[:size])
    e = (model.x[size:])
    #mu = find_mu(g0, beta, wn, V, e, mu, shift, erabs)
    return abs(V), e, model.fun

@numba.jit(fastmath=True)
def diff(par, g0, wn, mu):
    '''
    Рассчитывает разность между извесной гибридизационной функцией (g0 ** -1 - 1j*wn)
    и фиттируемой функции delta с произвольными начальными параметрами c учетом веса weight = 1 / wn
    '''
    delta = np.zeros(len(wn), dtype=np.complex64) 
    edge = len(par)//2
    V = par[:edge]
    e = par[edge:]
    for i in range(edge):
        delta -= V[i] ** 2 / (1j * wn - e[i])
    return np.sum(( abs((g0 ** -1).imag*1j  - 1j*wn - 1j*delta.imag)**2 + abs((g0 ** -1).real - mu  - delta.real)**2 )/ wn ** 2)

def _func(wn, V, e, mu):
    res = np.zeros(len(wn), dtype=np.complex128)
    for i in range(len(V)):
        res -= (V[i]) ** 2 / (1j * wn - e[i])
    return (res + mu + 1.0j * wn) ** -1

import numpy as np
from scipy import integrate
import numba 
import multiprocessing as mp
from itertools import product 

def dispersion(name, points):
    '''
    Construct dispersion read from file "name"
    '''
    assert all(points)
    nkx, nky, nkz = points
    with open(name) as f:
        data = [i for i in f.read().split('\n') if i != '']

    vectors = np.array([eval(i) for i in data[0].replace(' ', '').replace(')(', ') (').split(' ') ], dtype=np.float64)
    N = np.int32(data[1])
    coordinates = np.array([eval(i) for i in data[2].replace(' ', '').replace(')(', ') (').split(' ') ], dtype=np.float64)

    R = []
    T = {}
    for k in data[3:]:
        parts = k.split(')')
        r = eval(parts[0] + ')')
        i,j,t = map(float, parts[1].split())
        i = np.int(i)
        j = np.int(j)
        if r not in T.keys():
            R = np.append(R, r)
            T[r] = np.zeros((N,N), dtype = np.float64)
        T[r][i,j] += t
    R = R.reshape((-1, 3))

    index = range(3)
    for i in index:
        if all(vectors[i] == np.zeros(3)): 
            ind = index.pop(i)
    k_vec = np.zeros((3,3), dtype = np.float64)
    vs = np.zeros((3,3), dtype = np.float64)
    vs[index] = vectors[index]
    if len(index) == 1:
        V = np.linalg.norm(vs[index])
        k_vec = vs / V**2
    elif len(index) == 2:
        V = np.linalg.norm(np.cross(vs[index[0]], vs[index[1]]))
        n = np.cross(vs[index[0]], vs[index[1]]) / V
        if ind == 1:
            n *= -1
        vs[ind] = n
        for i, j in enumerate([[1,2], [2,0], [0, 1]]):
             k_vec[i] = np.cross(vs[j[0]],vs[j[1]]) / V
        k_vec[ind] = 0.
    else:
        V = np.dot(vs[0], np.cross(vs[1], vs[2]))
        for i, j in enumerate([[1,2], [2,0], [0, 1]]):
             k_vec[i] = np.cross(vs[j[0]],vs[j[1]]) / V

    E = np.zeros((nkx*nky*nkz,N), dtype = np.complex128)
    ni = product(range(1,nkx + 1),range(1, nky + 1),range(1, nkz + 1))
    Ni  = np.array([nkx,nky,nkz])

    k_points = []
    for n in ni:
        k_points.append(np.sum((2. * np.array(n) - Ni - 1) * k_vec / 2. / Ni , axis = 0))

    for k in range(len(k_points)):
        H = np.zeros((N,N), dtype = np.complex128)
        for r in R:
            for i in range(N):
                for j in range(N):
                    H[i,j] += T[tuple(r)][i,j] * np.exp(2. * np.pi * 1j * np.dot(k_points[k] , coordinates[i] - coordinates[j] + r))
        E[k] = np.linalg.eig(H)[0]
    return E.T


@numba.jit(nopython=True)
def g0_real(disp, wn, sigma, mu):
    return (mu - disp - sigma.real) / ((mu - disp - sigma.real) ** 2 + (-wn + sigma.imag) ** 2)


@numba.jit(nopython=True)
def g0_imag(disp, wn, sigma, mu):
    return (-wn + sigma.imag) / ((mu - disp - sigma.real) ** 2 + (-wn + sigma.imag) ** 2)


def f(funcs, args, x, opt, _dict, k): 
    return custom_integrate(funcs, args, x, opt, _dict, k)


def custom_integrate(funcs, args, x, opt, dct, k):
    real = integrate.nquad(funcs[0], [x] * 3, args=args, opts=opt)
    imag = integrate.nquad(funcs[1], [x] * 3, args=args, opts=opt)
    dct[k] = real[0] + 1j * imag[0]
    return dct


def g_imp(disp, wn, num_iw, sigma=0, mu=0, limit=50):
    k = 0
    g0 = np.zeros(num_iw, dtype=np.complex128)
    if  type(sigma) == int:
        sigma = np.zeros(num_iw, dtype=np.complex128)
    V = 8 * np.pi**3
    opt = {'limit' : limit, 'epsrel' : 1e-8, 'epsabs' : 1e-8}
    x = [-np.pi, np.pi]
    m = mp.Manager()
    dct = m.dict()
    p = [
        mp.Process(target=f, args=([g0_real, g0_imag], (disp, wn[k], sigma[k], mu), x, opt, dct, k))
        for k in range(mp.cpu_count() - 1)
        ]
    for i in p:
        i.start()
    while k < num_iw:
        for i in range(mp.cpu_count() - 1):
            if not p[i].is_alive() and k < num_iw:
                p[i] = mp.Process(target=f, args=([g0_real, g0_imag], (disp, wn[k], sigma[k], mu), x, opt, dct, k))
                p[i].start()
                k += 1
    for i in p:
        i.join()
    for k in range(num_iw):
        g0[k] = dct[k]
    return g0 / V
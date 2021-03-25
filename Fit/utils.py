import numpy as np
import h5py
import time
from numba import njit, prange
import scipy.linalg
import jax.numpy as jnp

def fake_templates(nbins,
                   nsignal=100,
                   nbkgd0=400,
                   nbkgd1=200,
                   seed=None):
    np.random.seed(seed)
    
    s = -1 * np.random.exponential(0.1, nsignal) + 1
    signal, bins = np.histogram(s, nbins, range=(0,1))


    b0 = np.random.exponential(0.1, nbkgd0)
    bkgd0, _ = np.histogram(b0, nbins, range=(0,1))

    b1 = np.random.normal(0.5, .25, nbkgd1)
    bkgd1, _ = np.histogram(b1, nbins, range=(0,1))

    return signal, bkgd0, bkgd1, bins

def poisson_multiverse(hist, nuniverses=100):
    return np.random.poisson(hist, size=(nuniverses, len(hist)))

def plot_templates(templates,
                   labels,
                   bins,
                   xlabel='Template Bins',
                   name=None,
                   **kwargs):
    import matplotlib.pyplot as plt
    for temp, label in zip(templates, labels):
        plt.hist(bins[:-1], weights=temp, bins=bins,
                 label=label,
                 histtype='stepfilled',
                 alpha=0.7,
                 **kwargs)

    plt.xlabel(xlabel)
    plt.ylabel('Events')
    plt.legend(loc='best')
    plt.show()

    if name:
        plt.savefig(name)
        print(f'Wrote {name}')

def fake_mv(nbins=10,nuniverses=100):
    return np.random.rand(nuniverses, nbins)

def cov_inv(mv):
    cov = mv_covariance(mv)
    u,s,v = np.linalg.svd(cov)
    inv = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
    return cov, inv
    

@njit(cache=True, parallel=True)
def mv_covariance(mv):
    nbins = len(mv[0])
    cov = np.zeros((nbins, nbins))
    n = len(mv)
    for i in prange(nbins):
        for j in prange(nbins):
            for imv in prange(n):
                cov[i][j] += (mv[imv][i] - mv[0][i]) * (mv[imv][j] - mv[0][j])

    cov /= (n - 1)

    return cov

# precompile mv_covariance with a small matrix
mv_covariance(fake_mv(2, 10))

def plot_mat(cov, name=None, **kwargs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    heat = ax.matshow(cov,
                      cmap=plt.get_cmap('Purples'),
                      **kwargs)
    plt.colorbar(heat, ax=ax)
    plt.show()
    if name:
        plt.savefig(name)
        print(f'Writing {name}')
                     



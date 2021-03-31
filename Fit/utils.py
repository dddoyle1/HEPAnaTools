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
                   seed=None,
                   shape=None):
    np.random.seed(seed)

    
    try:
        iter(shape)
        nsignal = (*shape, nsignal)
        nbkgd0  = (*shape, nbkgd0 )
        nbkgd1  = (*shape, nbkgd1 )

    except TypeError:
        pass


    s = -1 * np.random.exponential(0.1, nsignal) + 1
    b0 = np.random.exponential(0.1, nbkgd0)
    b1 = np.random.normal(0.5, .25, nbkgd1)
    
    multi_dim = len(s.shape) > 1
    # flatten all but last axis so we can make separate histograms    
    if multi_dim:
        signal = np.array([np.histogram(hs , nbins, range=(0, 1)) for hs  in s .reshape(np.prod(s .shape[:-1]), s .shape[-1])])
        bkgd0  = np.array([np.histogram(hb0, nbins, range=(0, 1)) for hb0 in b0.reshape(np.prod(b0.shape[:-1]), b0.shape[-1])])
        bkgd1  = np.array([np.histogram(hb1, nbins, range=(0, 1)) for hb1 in b1.reshape(np.prod(b1.shape[:-1]), b1.shape[-1])])

        bins   = np.asarray(signal[:,1].reshape(s .shape[:-1]).tolist())
        signal = np.asarray(signal[:,0].reshape(s .shape[:-1]).tolist())
        bkgd0  = np.asarray(bkgd0 [:,0].reshape(b0.shape[:-1]).tolist())
        bkgd1  = np.asarray(bkgd1 [:,0].reshape(b1.shape[:-1]).tolist())
    
    else:
        signal, bins = np.histogram(s, nbins, range=(0,1))
        bkgd0, _ = np.histogram(b0, nbins, range=(0,1))
        bkgd1, _ = np.histogram(b1, nbins, range=(0,1))

    return signal, bkgd0, bkgd1, bins

def poisson_multiverse(hist, nuniverses=100):
    return np.random.poisson(hist, size=(nuniverses, len(hist)))

def plot_templates(templates,
                   labels,
                   bins,
                   xlabel='Template Bins',
                   name=None,
                   data=None,
                   yerr=None,
                   scales=None,
                   chisq=None):
    import matplotlib.pyplot as plt

    if scales and len(scales) != len(templates):
        print(f'Scales ({scales.shape}) not compatible with templates ({templates.shape})')
        exit
    if scales is None: scales = np.ones(len(templates))
    
    for temp, scale, label in zip(templates, scales, labels):
        plt.hist(bins[:-1], weights=np.dot(temp, scale), bins=bins,
                 label=label,
                 histtype='stepfilled',
                 alpha=0.7)

    if data is not None:
        width = (bins[1:] - bins[:-1]) / 2
        x = bins[:-1] + width
        plt.errorbar(x, data, #xerr=width, #yerr=yerr,                     
                     marker='.',
                     ls='none',
                     mfc='black',
                     color='black',
                     label='Data')


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
                     



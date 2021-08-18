from hepanatools.shift.hist import Hist1D, Hist2D
from hepanatools.utils.math import chisq
import scipy.optimize    
import numba
import numpy as np
from functools import partial

class Bounds:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        if x > self.ub: return self.ub
        if x < self.lb: return self.lb
        return x

class PassThrough:
    def __init__(self):
        pass

    def __call__(self, x):
        return x
    
class PDF2D(Hist2D):
    def __init__(self,
                 nominal,
                 shifted,
                 xbins,
                 ybins,
                 bins_func=None,
                 **kwargs):
        if callable(bins_func): _, _, xbins, ybins = bins_func(nominal, shifted, xbins, ybins)

        super().__init__(nominal,
                         shifted - nominal,
                         bins=(xbins,ybins),
                         density=True,
                         **kwargs)

class CDF2D(Hist2D):
    def __init__(self, *args, pdf=None, constraint=PassThrough(), **kwargs):
        if pdf is None: pdf = PDF2D(*args, **kwargs)
        
        self.n = CDF2D._from_pdf(pdf.n, pdf.yaxis.edges)
        
        self.xaxis = pdf.xaxis
        self.yaxis = pdf.yaxis
        self.constraint = constraint

    @staticmethod
    @numba.jit(nopython=True)
    def _from_pdf(pdf, yedges):
        n = np.zeros_like(pdf)
        deltay = np.diff(yedges)
        for i in np.arange(pdf.shape[0]):
            p = pdf[i,:]*deltay
            p = p / p.sum()
            c = np.cumsum(p)
            n[i,:] = c
        return n
        
    def Sample(self, x):
        ix = self.xaxis.FindBin(x)
        return CDF2D._sample1d(self.n[ix, :], self.yaxis.edges)

    def Shift(self, x):
        return self.constraint(self.Sample(x) + x)

    @numba.jit(nopython=True)
    def _sample1d(cdf, bins):
        y = np.random.uniform(cdf[0], cdf[-1])
        ub = min(cdf.shape[0]-1, np.searchsorted(cdf, y))

        b1 = ub
        b0 = ub - 1
        y0 = cdf[b0]
        y1 = cdf[b1]

        x0 = bins[b0+1]
        x1 = bins[b1+1]
        return x0 + (y - y0) * (x1 - x0) / (y1 - y0)

    
def ybins1(nominal, shifted, xbins, ybins):
    """
    Makes a bin range for the y (CDF/PDF) axis
    whose range is +/- 5 standard deviations from the mean 
    absolute shift distribution to remove significant outliers.

    returns:
    nominal -- unmodified array of nominal values
    shifted -- unmodified array of shifted values
    xbins -- unmodified input xbins
    ybins -- ybins+1 array of bin edges
    """
    abs_shifts = shifted - nominal
    mean = abs_shifts.mean()
    std = abs_shifts.std()
    return nominal, shifted, xbins, np.linspace(mean - 5 * std,
                                                mean + 5 * std,
                                                ybins+1)

def xbins_equal_prob(nominal, shifted, xbins, ybins):
    """
    Makes a bin range for the x (conditional) axis
    whose bin widths attempt to equalize frequentist probability
    of an event falling within each bin based on proportions.

    1. An array of equal proportions is determined for xbins
    2. Input nominal array is sorted, and entries corresponding with
       the proportions array are found.
       2.5.  If it is found that same-valued events span 
             multiple proportion intervals, these proportion intervals
             are removed, and user is warned             
    3. Edges are determined as the values of those events corresponding
       to the proportions
    

    returns:
    nominal -- unmodified array of nominal values
    shifted -- unmodified array of shifted values
    xbins -- (at most) xbins+1 array of bin edges
    ybins -- unmodified input ybins
    """
    sorted_nominal = np.sort(nominal)
    props = np.linspace(0, 1, xbins+1)
    xedges = sorted_nominal[np.array(sorted_nominal.shape[0] * props, dtype=int)[:-1]]
    xedges = np.append(xedges, sorted_nominal[-1])
    xedges, counts = np.unique(xedges, return_index=True)
    if (counts > 1).any():
        print(f'Warning: nbins changed from {xbins} --> {xedges.shape[0]}')
    xedges[-1] += 0.0001
    return nominal, shifted, xedges, ybins

class BinOptimizer:
    DEFAULT_MINIMIZER_OPTS = {'initial_constr_penalty': 1000,
                              'finite_diff_rel_step': 0.001}    
    def __init__(self,
                 nbins,
                 range,
                 axis=0,
                 analytic_jacobian=True,
                 minimizer_opts=None):
        self.axis = axis
        self.nbins = nbins
        self.bins = [[], []]
        self.bins[axis] = np.linspace(*range, self.nbins+1)
        
        self.results = None
        self.minimizer_opts = BinOptimizer.DEFAULT_MINIMIZER_OPTS
        self.minimizer_opts.update(minimizer_opts)

        jac = partial(self._jac, lb=range[0], ub=range[0]) if analytic_jacobian else '2-point'
        
        self.constraint = scipy.optimize.NonlinearConstraint(partial(BinOptimizer._c,
                                                                     lb=range[0],
                                                                     ub=range[1]),
                                                             np.zeros(self.nbins - 1),
                                                             np.ones(self.nbins - 1),
                                                             jac=jac)

        self.best_chisq = 1e10
        self.best_bins = None

    def __call__(self, fun, xbins, ybins,*, x0=None, **kwargs):
        if x0 is not None: self.bins[self.axis] = x0
        if self.axis == 0: self.bins[1] = ybins
        if self.axis == 1: self.bins[0] = xbins

        self.results = scipy.optimize.minimize(self._funwrap,
                                               self.bins[self.axis][1:-1],
                                               args=(fun),
                                               method='trust-constr',
                                               constraints=[self.constraint],
                                               options=self.minimizer_opts,
                                               **kwargs)
        
        self._update(self.results.x)
        return self.results
        
    def _update(self, floating_bins):
        self.bins[self.axis][1:-1] = floating_bins

    def _funwrap(self, x, fun, *args, **kwargs):
        self._update(x)
        chi = fun(*args, **kwargs)
        
        # save the best fit in case the minimizer gets taken out of
        # minima and can't find it's way back in
        if chi < self.best_chisq:
            self.best_chisq = chi
            self.best_bins = self.bins[self.axis]
        return chi

    @staticmethod
    @numba.jit(nopython=True)    
    def _c(x, lb, ub):
        c = np.zeros(len(x))
        c[0 ] = (x[0] - lb) / (x[1] - lb)
        c[-1] = (x[-1] - x[-2] ) / (ub - x[-2])
        for i in range(1,len(x)-1):
            c[i] = (x[i] - x[i-1]) / (x[i+1] - x[i-1])
        return c
    
    @staticmethod
    @numba.jit(nopython=True)
    def _jac(x, lb, ub):
        jac = np.zeros((len(x), len(x)))
        jac[0,0] = 1 / (x[1] - lb)
        jac[0,1] = (lb - x[0]) / (x[1] - lb)**2
        jac[1,0] = jac[0,1]

        jac[-1,-1] = 1 / (ub - x[-2])
        for i in range(1, len(x)-1):
            #\delta_{i,j}
            jac[i,i] = 1 / (x[i+1] - x[i-1])

            #\delta_{i+1,j}
            jac[i, i+1] = (x[i-1] - x[i]) / (x[i+1] - x[i-1])**2
            jac[i+1, i] = jac[i, i+1]

        return jac

class TestBinOptimizer(BinOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fun(self, X, target_hist):
        u, _ = np.histogram(X, bins=self.bins[self.axis])
        return chisq(target_hist, u)

    def __call__(self, X, target_hist, **kwargs):
        super().__call__(partial(self.fun, X=X, target_hist=target_hist),
                         xbins=None, ybins=None,
                         **kwargs)
        return self.results
        
        
class CDFBinOptimizer(BinOptimizer):
    def __init__(self,
                 obj_bins,
                 cdf_factory=CDF2D,
                 **kwargs):
        super().__init__(axis=0, **kwargs) 
        self.obj_bins = obj_bins
        
        self.cdf_factory = cdf_factory

    @staticmethod
    def FromConfig(config):
        return CDFBinOptimizer(config.objective_bins,
                               cdf_factory=partial(CDF2D, constraint=config.bounds),
                               nbins=config.xbins,
                               range=config.xlim,
                               minimizer_opts=config.minimizer_opts)


        
    def __call__(self, nominal, shifted, xbins, ybins, **kwargs):
        super().__call__(partial(self.fun,
                                 nominal=nominal,
                                 target=shifted),
                         xbins, ybins,
                         **kwargs)
        return nominal, shifted, self.bins[0], self.bins[1]         
    
    def fun(self, nominal, target):
        try:
            cdf = self.cdf_factory(nominal, target, xbins=self.bins[0], ybins=self.bins[1])
        except ValueError as err:
            print('Error: somehow fell into a non-monotonic parameter space')
            print(self.bins[self.axis])
            raise err
            
        
        hshifted = Hist1D(np.array([cdf.Shift(x) for x in nominal]), bins=self.obj_bins)
        htarget  = Hist1D(target, bins=self.obj_bins)
        
        return chisq(htarget.n, hshifted.n)


class Callback:
    def __init__(self, report_every=5):
        self.nfcn = 0
        self._report_every = report_every
        
class VerbosityCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self, xk,  state):
        if not self.nfcn % self._report_every:
            print(self.nfcn, state.fun, xk)
        self.nfcn += 1
        return False

class ProgressTrackerCallback(Callback):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.params = []
        self.fun_vals = []
        self.jac_vals = []
        self.fun_calls = []
        self.verbose = verbose
                              
    def __call__(self, xk,  state):
        if not self.nfcn % self._report_every:
            self.params.append(xk)
            self.fun_calls.append(self.nfcn)
            self.fun_vals.append(state.fun)
            self.jac_vals.append(state.jac)
            if self.verbose: print(self.nfcn, state.fun, xk)
        self.nfcn += 1
        return False

    def Draw(self, lb, ub, ax=None, cmap='plasma'):
        import matplotlib
        import matplotlib.pyplot as plt
        cmap = matplotlib.cm.get_cmap(cmap)
        
        if ax is None: ax = plt.gca()
        min_fun = min(self.fun_vals)
        max_fun = max(self.fun_vals)
        ones = np.ones(len(self.params[0])+2)
        for i in range(len(self.fun_calls)):
            x = np.concatenate(([lb], self.params[i], [ub]))
            y = self.fun_calls[i] * ones
            if max_fun - min_fun > 0:
                color_val = (self.fun_vals[i] - min_fun) / (max_fun - min_fun)
            else:
                color_val = 0.5
            ax.plot(x, y,
                    color=cmap(color_val),
                    marker='|',
                    ms=8, markeredgewidth=4)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_fun, vmax=max_fun))        
        cb = plt.colorbar(sm, ax=ax)
        cb.set_label(r'$\chi^2$')
        
        ax.set_yticks(self.fun_calls[::10])
        ax.set_xlim([lb, ub])
        ax.set_ylabel('Iterations')

        return ax        

from hepanatools.shift.hist import Hist1D, Hist2D, FileNameOrHandle
from hepanatools.utils.math import chisq
import scipy.optimize
import numba
import numpy as np
from functools import partial
import json
from mpi4py import MPI
import sys


class Bounds:
    """Simple bounds with upper and lower limits."""

    def __init__(self, lb: float, ub: float):
        self.lb = lb
        self.ub = ub

    def __call__(self, x: float) -> float:
        """
        If x is less than the upper bound and greater than the lower bound,
        return x.
        If x is greater than upper bound, return upper bound.
        If x is less than lower bound, return lower bound.
        """
        if x > self.ub:
            return self.ub
        if x < self.lb:
            return self.lb
        return x


class PassThrough:
    """No-op object."""

    def __init__(self):
        pass

    def __call__(self, x):
        """No-op"""
        return x


class PDF2D(Hist2D):
    def __init__(self, nominal, shifted, xbins, ybins, bins_func=None, **kwargs):
        if callable(bins_func):
            _, _, xbins, ybins = bins_func(nominal, shifted, xbins, ybins)

        super().__init__(
            nominal, shifted - nominal, bins=(xbins, ybins), density=True, **kwargs
        )


class CDF2D(Hist2D):
    def __init__(
        self,
        *args,
        pdf=None,
        constraint=PassThrough(),
        n=None,
        xaxis=None,
        yaxis=None,
        **kwargs,
    ):
        self.Shift = np.vectorize(
            self._sshift,
            otypes=[float],
            doc="Get shifted value given the nominal value by sampling the CDF",
        )
        self.Sample = np.vectorize(
            self._ssample,
            otypes=[float],
            doc="Sample the CDF to get a value of absolute change in the input variable",
        )
        if n is not None:
            self.n = n
            self.xaxis = xaxis
            self.yaxis = yaxis

        else:
            if pdf is None:
                pdf = PDF2D(*args, **kwargs)

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
            p = pdf[i, :] * deltay
            p = p / p.sum()
            c = np.cumsum(p)
            n[i, :] = c
        return n

    @staticmethod
    def FromH5(file_name_or_handle, path, constraint=PassThrough()):
        h = Hist2D.FromH5(file_name_or_handle, path)
        cdf = CDF2D(n=h.n, xaxis=h.xaxis, yaxis=h.yaxis, constraint=constraint)
        return cdf

    def _ssample(self, x):
        """To be wrapped with np.vectorize"""

        ix = self.xaxis.FindBin(x)
        return CDF2D._sample1d(self.n[ix, :], self.yaxis.edges)

    def _sshift(self, x):
        """To be wrapped with np.vectorize"""
        return self.constraint(self._ssample(x) + x)

    @numba.jit(nopython=True)
    def _sample1d(cdf, bins):
        y = np.random.uniform(cdf[0], cdf[-1])
        ub = min(cdf.shape[0] - 1, np.searchsorted(cdf, y))

        b1 = ub
        b0 = ub - 1
        y0 = cdf[b0]
        y1 = cdf[b1]

        x0 = bins[b0 + 1]
        x1 = bins[b1 + 1]
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
    return (
        nominal,
        shifted,
        xbins,
        np.linspace(mean - 5 * std, mean + 5 * std, ybins + 1),
    )


def ybins1_symmetric(nominal, shifted, xbins, ybins):
    """
    Symmeterized version of ybins1
    """
    abs_shifts = shifted - nominal
    mean = abs_shifts.mean()
    std = abs_shifts.std()
    return (
        nominal,
        shifted,
        xbins,
        np.concatenate(
            (
                np.linspace(mean - 5 * std, 0, int(ybins / 2) + 1),
                np.linspace(0, mean + 5 * std, int(ybins / 2) + 1)[1:],
            )
        ),
    )


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
    props = np.linspace(0, 1, xbins + 1)
    xedges = sorted_nominal[np.array(sorted_nominal.shape[0] * props, dtype=int)[:-1]]
    xedges = np.append(xedges, sorted_nominal[-1])
    xedges, counts = np.unique(xedges, return_index=True)
    if (counts > 1).any():
        print(f"Warning: nbins changed from {xbins} --> {xedges.shape[0]}")
    xedges[-1] += 0.0001
    return nominal, shifted, xedges, ybins


class BinOptimizer:
    DEFAULT_MINIMIZER_OPTS = {
        "initial_constr_penalty": 1000,
        "finite_diff_rel_step": 0.001,
    }

    def __init__(
        self,
        nbins,
        range,
        axis=0,
        analytic_jacobian=True,
        minimizer_opts=dict(),
        pad=0,
        exaggerate_jac=1,
        noise_scale=0.1,
    ):
        self.axis = axis
        self.nbins = nbins
        self.bins = [[], []]

        # here's the initial guess. Evenly spaced bins
        self.bins[axis] = np.linspace(*range, self.nbins + 1)

        self.results = None
        self.minimizer_opts = BinOptimizer.DEFAULT_MINIMIZER_OPTS
        self.minimizer_opts.update(minimizer_opts)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Minimizer options")
            print(json.dumps(self.minimizer_opts, indent=4))
            sys.stdout.flush()

        self.pad = (range[1] - range[0]) * pad
        self.exaggerate_jac = exaggerate_jac
        self.noise_scale = (
            self.bins[self.axis][-1] - self.bins[self.axis][0]
        ) * noise_scale

        jac = (
            partial(self._jac, lb=range[0], ub=range[0], exagg=self.exaggerate_jac)
            if analytic_jacobian
            else "3-point"
        )
        finite_diff_jac_sparsity = None

        if type(jac) is str:
            finite_diff_jac_sparsity = np.identity(self.nbins - 1)
            finite_diff_jac_sparsity[:-1, 1:] += np.identity(self.nbins - 2)
            finite_diff_jac_sparsity[1:, :-1] += np.identity(self.nbins - 2)

        self.constraint = scipy.optimize.NonlinearConstraint(
            partial(BinOptimizer._c, lb=range[0], ub=range[1]),
            np.zeros(self.nbins - 1) + self.pad,
            np.ones(self.nbins - 1) * (1 - self.pad),
            jac=jac,
            finite_diff_jac_sparsity=finite_diff_jac_sparsity,
        )

        self.best_chisq = 1e10
        self.best_bins = None

    def __call__(self, fun, xbins, ybins, *, x0=None, **kwargs):
        if x0 is not None:
            self.bins[self.axis] = x0
        if self.axis == 0:
            self.bins[1] = ybins
        if self.axis == 1:
            self.bins[0] = xbins

        self.results = scipy.optimize.minimize(
            self._funwrap,
            self.bins[self.axis][1:-1],
            args=(fun),
            method="trust-constr",
            constraints=[self.constraint],
            options=self.minimizer_opts,
            **kwargs,
        )

        self._update(self.results.x)
        return self.results

    def _add_noise(self, scale):
        self._update(np.sort(np.random.normal(self.bins[self.axis][1:-1], scale)))

    def _random_seed(self):
        self._update(
            np.sort(
                np.random.uniform(
                    self.bins[self.axis][0],
                    self.bins[self.axis][-1],
                    size=len(self.bins[self.axis]) - 2,
                )
            )
        )

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
        c[0] = (x[0] - lb) / (x[1] - lb)
        c[-1] = (x[-1] - x[-2]) / (ub - x[-2])
        for i in range(1, len(x) - 1):
            c[i] = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        return c

    @staticmethod
    @numba.jit(nopython=True)
    def _jac(x, lb, ub, exagg):
        jac = np.zeros((len(x), len(x)))
        jac[0, 0] = 1 / (x[1] - lb)
        jac[0, 1] = (lb - x[0]) / (x[1] - lb) ** 2

        jac[-1, -1] = 1 / (ub - x[-2])
        jac[-1, -2] = (x[-1] - x[-2]) / (ub - x[-2]) ** 2
        for i in range(1, len(x) - 1):
            # \delta_{i,j}
            jac[i, i] = 1 / (x[i + 1] - x[i - 1])

            # \delta_{i+1,j}
            jac[i, i + 1] = (x[i - 1] - x[i]) / (x[i + 1] - x[i - 1]) ** 2

            # \delta_{i-1, j}
            jac[i, i - 1] = jac[i, i + 1]

        return jac * exagg


class TestBinOptimizer(BinOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fun(self, X, target_hist):
        u, _ = np.histogram(X, bins=self.bins[self.axis])
        return chisq(target_hist, u)

    def __call__(self, X, target_hist, **kwargs):
        super().__call__(
            partial(self.fun, X=X, target_hist=target_hist),
            xbins=None,
            ybins=None,
            **kwargs,
        )
        return self.results


class CDFBinOptimizer(BinOptimizer):
    def __init__(
        self,
        obj_bins,
        cdf_factory=CDF2D,
        retries=0,
        nquick_seeds=1,
        nmultistarts=1,
        **kwargs,
    ):
        super().__init__(axis=0, **kwargs)
        self.obj_bins = obj_bins

        self.cdf_factory = cdf_factory
        self.retries = retries
        self.nquick_seeds = nquick_seeds
        self.nmultistarts = nmultistarts
        self.target = None

    @staticmethod
    def FromConfig(config):
        return CDFBinOptimizer(
            config.objective_bins,
            cdf_factory=partial(CDF2D, constraint=config.bounds),
            nbins=config.xbins,
            range=config.xlim,
            minimizer_opts=config.minimizer_opts,
            pad=config.pad,
            noise_scale=config.noise_scale,
            exaggerate_jac=config.exaggerate_jac,
            retries=config.retries,
            analytic_jacobian=config.analytic_jacobian,
            nquick_seeds=config.nquick_seeds,
            nmultistarts=config.nmultistarts,
        )

    def __call__(self, nominal, shifted, xbins, ybins, **kwargs):
        self.target = Hist1D(shifted - nominal, self.obj_bins)
        nseeds = max(self.nquick_seeds, self.nmultistarts)

        # generate random seed bins from a uniform distribution
        seeds = np.array(
            [
                np.sort(
                    np.random.uniform(
                        self.bins[0][0], self.bins[0][-1], self.bins[0].shape[0] - 2
                    )
                )
                for _ in range(nseeds)
            ]
        )

        self.bins[1] = ybins

        seed_chi2 = []
        for iseed in range(nseeds):
            self.bins[0][1:-1] = seeds[iseed]
            seed_chi2.append(self.fun(nominal, shifted))

        # collect seed results from all ranks
        comm = MPI.COMM_WORLD
        seed_chi2 = comm.gather(seed_chi2, root=0)
        seeds = comm.gather(seeds, root=0)

        # sort among all ranks and report results
        if comm.Get_rank() == 0:
            seeds = np.concatenate(seeds)
            seed_chi2 = np.concatenate(seed_chi2)
            sorted_idx = np.argsort(seed_chi2).astype(int)
            seeds = seeds[sorted_idx]
            seed_chi2 = seed_chi2[sorted_idx]
            print("".join(["-"] * 40))
            print("Quick seed results:")
            for n in range(nseeds):
                print(n + 1, "%.3f" % seed_chi2[n], seeds[n])
            print("".join(["-"] * 40))
            sys.stdout.flush()
        else:
            seeds = np.empty((comm.Get_size() * nseeds, self.bins[0].shape[0] - 2,))
            seed_chi2 = np.empty(comm.Get_size() * nseeds,)
        # broadcast from rank one, and have each rank work on a different piece
        comm.Bcast(seed_chi2, root=0)
        comm.Bcast(seeds, root=0)

        offset = comm.Get_rank()
        stride = comm.Get_size()
        nseeds_to_do = (
            int(self.nmultistarts / comm.Get_size())
            + self.nmultistarts % comm.Get_size()
        )
        for start in range(nseeds_to_do):
            self.bins[0][1:-1] = seeds[offset::stride][start]

            # It's possible for the fitter to disobey the constraint
            # that bins must be monotonically increasing.
            # If that happens, catch it and add a little bit of noise
            # to this start's seed so hopefully we don't fall down the same
            # rabbit hole.
            for retry in range(self.retries):
                try:
                    super().__call__(
                        partial(self.fun, nominal=nominal, shifted=shifted),
                        xbins,
                        ybins,
                        **kwargs,
                    )

                except ValueError as err:
                    print(
                        f"Error: Fell into a non-monotonic parameter space. Retry {retry+1} / {self.retries}"
                    )
                    self._update(seeds[start])
                    self._add_noise(scale=self.noise_scale * 2 ** (retry + 1))

                    # if this does happen, go to the top of the inner loop
                    continue

                # if the fit succeeds, break the inner loop
                break

        assert np.array_equal(
            self.best_bins, np.sort(self.best_bins)
        ), "Error: Fit did not yield a valid set of bins."

        nbins = self.bins[0].shape[0]
        # collect best fit results from all ranks
        best_bins = comm.gather(self.best_bins, root=0)
        best_chisqs = comm.gather(self.best_chisq, root=0)

        # Have rank 0 find the best fit of all ranks and print results
        if comm.Get_rank() == 0:
            best_rank = np.argsort(best_chisqs).astype(int)[0]
            self.bins[0] = best_bins[best_rank]
            best_chisq_buffer = np.array([best_chisqs[best_rank]])
            print("Results")
            print("chisq = %.2f" % self.best_chisq)
            print("bins  = ", self.bins[0])
            sys.stdout.flush()
        else:
            self.bins[0] = np.empty(nbins)
            best_chisq_buffer = np.empty(1)

        # broadcast best fit from rank 0
        comm.Bcast(self.bins[0], root=0)
        comm.Bcast(best_chisq_buffer, root=0)

        self.best_chisq = best_chisq_buffer[0]

        return nominal, shifted, self.bins[0], self.bins[1]

    def fun(self, nominal, shifted):
        cdf = self.cdf_factory(nominal, shifted, xbins=self.bins[0], ybins=self.bins[1])
        hshifted = Hist1D(cdf.Sample(nominal), bins=self.obj_bins)
        return chisq(self.target.n, hshifted.n)


class BruteResult:
    def __init__(self, fun, jac=None):
        self.fun = fun
        self.jac = jac


class Callback:
    def __init__(self, report_every=5):
        self.nfcn = 0
        self._report_every = report_every

    def __call__(self, xk, state):
        pass


class BruteCDFBinOptimizer(CDFBinOptimizer):
    def __init__(self, *args, nsamples=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsamples = nsamples

    @staticmethod
    def FromConfig(config):
        return BruteCDFOptimizer(
            config.objective_bins,
            cdf_factory=partial(CDF2D, constraint=config.bounds),
            nbins=config.xbins,
            range=config.xlim,
            samples=config.brute_samples,
        )

    def __call__(self, nominal, shifted, xbins, ybins, callback=Callback()):
        self.bins[1] = ybins
        samples = np.sort(
            np.random.uniform(
                self.bins[0][0],
                self.bins[0][-1],
                (len(self.bins[0]) - 2, self.nsamples),
            ),
            axis=0,
        )
        fun_vals = []
        for isample in range(samples.shape[1]):
            self._update(samples[:, isample])
            fun_vals.append(self.fun(nominal, shifted))

        fun_vals = np.array(fun_vals)

        sorted_idx = np.argsort(fun_vals)[::-1]

        for idx in sorted_idx:
            callback(samples[:, idx], BruteResult(fun_vals[idx]))

        self._update(samples[:, sorted_idx[0]])
        return nominal, shifted, self.bins[0], self.bins[1]


class VerbosityCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, xk, state):
        if not self.nfcn % self._report_every:
            print(self.nfcn, state.fun, xk)
        self.nfcn += 1
        return False


class ProgressTrackerCallback(Callback):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.params = []
        self.fun_vals = []
        self.fun_calls = []
        self.verbose = verbose

    def __call__(self, xk, state):
        if not self.nfcn % self._report_every:
            self.params.append(xk)
            self.fun_calls.append(self.nfcn)
            self.fun_vals.append(state.fun)
            if self.verbose:
                print(
                    f"[{MPI.COMM_WORLD.Get_rank()}] {self.nfcn}  {state.fun:.2f}", xk,
                )
                sys.stdout.flush()
        self.nfcn += 1
        return False

    def ToH5(self, file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, "w") as f:
            g = f.create_group(path)
            g.create_dataset("params", data=self.params, compression="gzip")
            g.create_dataset("fun_vals", data=self.fun_vals, compression="gzip")
            g.create_dataset("fun_calls", data=self.fun_calls, compression="gzip")

    def FromH5(file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, "r") as f:
            g = f.get(path)
            params = g.get("params")[:]
            fun_vals = g.get("fun_vals")[:]
            fun_calls = g.get("fun_calls")[:]
        progress = ProgressTrackerCallback()
        progress.params = params
        progress.fun_vals = fun_vals
        progress.fun_calls = fun_calls
        return progress

    def Draw(self, lb, ub, ax=None, cmap="plasma"):
        import matplotlib
        import matplotlib.pyplot as plt

        cmap = matplotlib.cm.get_cmap(cmap)

        if ax is None:
            ax = plt.gca()
        min_fun = min(self.fun_vals)
        max_fun = max(self.fun_vals)
        ones = np.ones(len(self.params[0]) + 2)
        for i in range(len(self.fun_calls)):
            x = np.concatenate(([lb], self.params[i], [ub]))
            y = self.fun_calls[i] * ones
            if max_fun - min_fun > 0:
                color_val = (self.fun_vals[i] - min_fun) / (max_fun - min_fun)
            else:
                color_val = 0.5
            ax.plot(x, y, color=cmap(color_val), marker="|", ms=8, markeredgewidth=4)
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=min_fun, vmax=max_fun)
        )
        cb = plt.colorbar(sm, ax=ax)
        cb.set_label(r"$\chi^2$")

        # ax.set_yticks(self.fun_calls[np.arange(len(self.fun_calls),)])
        ax.set_xlim([lb, ub])
        ax.set_ylabel("Iterations")

        return ax

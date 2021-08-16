from hepanatools.shift.hist import Hist1D, Hist2D
import numba
import numpy as np

class PDF2D(Hist2D):
    def __init__(self, nominal, shifted, xbins, ybins, bins_func=None, **kwargs):
        if callable(bins_func): _, _, xbins, ybins = bins_func(nominal, shifted, xbins, ybins)

        super().__init__(nominal,
                         shifted - nominal,
                         bins=(xbins,ybins),
                         density=True,
                         **kwargs)

    def CDF(self):
        cdf = np.empty_like(self.n)
        deltay = np.diff(self.yaxis.edges)
        for i in np.arange(self.n.shape[0]):
            p = self.n[i,:]*deltay
            p = p / p.sum()
            c = np.cumsum(p)
            cdf[i,:] = c
        cdf = CDF2D.Filled(cdf, self.xaxis, self.yaxis)
        cdf.__class__ = CDF2D
        return cdf
        
class CDF2D(Hist2D):
    def __init__(self, *args, **kwargs):
        pdf = PDF2D(*args, **kwargs)
        self = pdf.CDF()

    def Sample(self, x):
        ix = self.xaxis.FindBin(x)
        return CDF2D._sample1d(self.n[ix, :], self.yaxis.edges)

    def Shift(self, x):
        return self.Sample(x) + x

    def ShiftBounded(self, x, lower, upper):
        s = self.Shift(x)
        if s > upper: s = upper
        if s < lower: s = lower
        return s

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
    abs_shifts = shifted - nominal
    mean = abs_shifts.mean()
    std = abs_shifts.std()
    return nominal, shifted, xbins, np.linspace(mean - 5 * std,
                                               mean + 5 * std,
                                               ybins+1)

def xbins_equal_prob(nominal, shifted, xbins, ybins):
    sorted_nominal = np.sort(nominal)
    props = np.linspace(0, 1, xbins+1)
    xedges = sorted_nominal[np.array(sorted_nominal.shape[0] * props, dtype=int)[:-1]]
    xedges = np.append(xedges, sorted_nominal[-1])
    xedges, counts = np.unique(xedges, return_index=True)
    if (counts > 1).any():
        print(f'Warning: nbins changed from {xbins} --> {xedges.shape[0]}')
    xedges[-1] += 0.0001
    return nominal, shifted, xedges, ybins

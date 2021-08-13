from HEPAnaTools.EventCDFShift import Hist, Hist2D
import numba

class PDF2D(Hist2D):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, density=True, **kwargs)

    def CDF(self):
        cdf = np.empty_like(self.n)
        deltay = np.diff(self.yaxis.bins)
        for i in np.arange(self.n.shape[0]):
            p = n[i,:]*deltay
            p = p / p.sum()
            c = np.cumsum(p)
            cdf[i,:] = c
        return CDF2D.Filled(cdf, self.xaxis, self.yaxis)
        
class CDF2D(Hist2D)
    def __init__(self, *args, **kwargs):
        pdf = PDF2D(*args, **kwargs)
        self = pdf.CDF()

    @numba.jit(nopython=True)
    def Sample(self, x):
        ix = self.xaxis.find_bin(x)
        y = np.random.uniform(self.n[ix,0], self.[ix,-1])

        ub = np.where(self.n[ix,:] >= y)[0][0]
        b1 = ub
        b0 = ub - 1
        y0 = self.n[ix, b0]
        y1 = self.n[ix, b1]

        x0 = self.yaxis.bins[b0+1]
        x1 = self.yaxis.bins[b1+1]

        return x0 + (y - y0) * (x1 - x0) / (y1 - y0)

    
        

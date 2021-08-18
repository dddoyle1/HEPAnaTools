from hepanatools.shift import *
import hepanatools.utils.plot as hpl
import matplotlib.pyplot as plt

def plot_sample1d(pdf, cdf, ibin, name, nsamples=10000):
    fig, ax = hpl.split_subplots(nrows=2, figsize=(10,8))

    samples = np.array([CDF2D._sample1d(cdf.n[ibin,:], cdf.yaxis.edges) for _ in range(nsamples)])

    # normalize pdf
    pdf1d = Hist1D.Filled(pdf.n[ibin,:] / (sum(pdf.n[ibin,:] * np.diff(pdf.yaxis.edges))),
                        pdf.yaxis)

    sampled = Hist1D(samples, pdf.yaxis.edges, density=True)
    ratio = Hist1D.Filled(sampled.n / pdf1d.n, pdf1d.xaxis)
        
    pdf1d.Draw(ax[0], histtype='step', color='r', label='PDF', hatch='//')
    sampled.Draw(ax[0], histtype='step', linewidth=2, color='b', label='CDF-Sampled', density=True)

    ax[0].legend()
    ax[1].axhline(1, linestyle='--', color='gray')
    ax[1].set_ylim([0.5, 1.5])
    
    ax[0].set_ylabel('Arbitrary Units')
    ax[1].set_ylabel('Sampled / PDF')
    ax[1].set_xlabel('Shift - Nominal')
    
    np.nan_to_num(ratio.n, 0)
    ratio.Draw(ax[1], histtype='step', linewidth=2, color='k')
    
    hpl.savefig(name)


    
    

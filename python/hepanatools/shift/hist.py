import numpy as np
import pandas as pd
import h5py
class FileNameOrHandle:
    """ 
    Wraps a file handle modify the context manager depending
    on when the file handle was created

    If file_name_or_handle is h5py.File, we assume the user
    is responsible for closing the file. Otherwise, we'll open and close 
    the file.
    """
    def __init__(self, file_name_or_handle, mode='r', **kwargs):
        self.delegate = True
        self.handle = file_name_or_handle
        if type(self.handle) is not h5py.File:
            self.delegate = False
            self.handle = h5py.File(self.handle, mode, **kwargs)

    def __enter__(self):
        return self.handle

    def __exit__(self):
        if not self.delegate: self.handle.close()

        
class Axis:
    def __init__(self, edges):
        self.edges = edges
    def NBins(self):
        return len(self.edges) - 1
    
    def FindBin(self, x, **kwargs):
        return np.digitize(x, self.edges, **kwargs) - 1

    def GetBinCenter(self, i):
        return (self.edges[i+1] + self.edges[i]) / 2


class Hist1D:
    def __init__(self, X, edges, **kwargs):
        self.n, edges = np.histogram(X, bins, **kwargs)
        self.xaxis = Axis(edges)

    @staticmethod
    def FromH5(file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'r') as f:
            dset = f.get(path)[:]
            h = Hist.FilledHist(dset[:], dset.attrs['xedges'])
        return h

    @staticmethod
    def Filled(n, edges):
        h = Hist1D([], [])
        h.n = n
        h.xaxis = edges if type(edges) is Axis else Axis(edges)
        return h
        

    def FindBin(self, x):
        return self.xaxis.FindBin(x)
    
    def ToH5(self, file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'w') as f:
            dset = f.create_dataset(path, data=self.n, compression='gzip')
            dset.attrs['xaxis'] = self.xaxis.edges

    def ToROOT(self, output, name):
        import ROOT
        from array import array
        h = ROOT.TH1D('', '', len(self.xaxis.edges), array('d', self.xaxis.edges))
        for i in range(len(self.xaxis.edges)):
            h.SetBinContent(i+1, self.n[i])
        h.Write(name)


    def Draw(self, ax, **kwargs):
        ax.hist(self.xaxis.edges[:-1], edges=self.xaxis.edges, weights=self.n, **kwargs)
        ax.set_xlim([self.xaxis.edges[0], self.xaxis.edges[-1]])        

class Hist2D:
    def __init__(self, X, Y, edges, **kwargs):
        self.n, xedges, yedges = np.histogram2d(X, Y, bins=edges, **kwargs)
        self.xaxis = Axis(xedges)
        self.yaxis = Axis(yedges)

    @staticmethod
    def FromH5(file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'r') as f:
            dset = f.get(path)[:]
            h = Hist2D.FilledHist(dset[:], dset.attrs['xedges'], dset.attrs['yedges'])
        return h

    @staticmethod
    def Filled(n, xedges, yedges):
        h = Hist2D([], [], [])
        h.n = n
        h.xaxis = xedges if type(xedges) is Axis else Axis(xedges)
        h.yaxis = yedges if type(yedges) is Axis else Axis(yedges)
        return h

    def FindBin(self, x, y, gid=False):
        ix = self.xaxis.FindBin(x)
        iy = self.yaxis.FindBin(y)
        if not gid:
            return ix, iy
        else:
            return ix * self.n.shape[1] + iy 

    def ToH5(self, file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'w') as f:
            dset = f.create_dataset(path, data=self.n, compression='gzip')
            dset.attrs['xaxis'] = self.xaxis.edges
            dset.attrs['yaxis'] = self.yaxis.edges
            
    def ToROOT(self, output, name):
        import ROOT
        from array import array
        h = ROOT.TH2D('', '',
                      len(self.xaxis.edges), array('d', self.xaxis.edges),
                      len(self.yaxis.edges), array('d', self.yaxis.edges))
        
        for i in range(len(self.xaxis.edges)):
            h.SetBinContent(i+1, self.n[i])
        h.Write(name)

    def ToHist1D(self):
        return Hist1D.Filled(self.n.flatten(),
                             np.linspace(0,
                                         np.prod(*self.n.shape),
                                         np.prod(*self.n.shape)+1))        
    
    def Draw(self, ax, colorbar=True,**kwargs):
        _X, _Y = np.meshgrid(self.xaxis.edges, self.yaxis)
        im = ax.pcolormash(_X, _Y, self.n.T, **kwargs)
        ax.set_xlim([self.xaxis.edges[0], self.xaxis.edges[-1]])
        ax.set_ylim([self.yaxis.edges[0], self.yaxis.edges[-1]])        
        if colorbar: plt.colorbar(im)

                                  
class Hist3D:
    def __init__(self, X, Y, Z, edges, **kwargs):
        self.n, edges = np.histogramdd((X, Y, Z),
                                       bins=edges,
                                       **kwargs)
        self.xaxis = Axis(edges[0])
        self.yaxis = Axis(edges[1])
        self.zaxis = Axis(edges[2])

    @staticmethod
    def FromH5(file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'r') as f:
            dset = f.get(path)[:]
            h = Hist3D.FilledHist(dset[:],
                                  dset.attrs['xedges'],
                                  dset.attrs['yedges'],
                                  dset.attrs['zedges'])
        return h

    @staticmethod
    def Filled(n, xedges, yedges, zedges):
        h = Hist3D([], [], [])
        h.n = n
        h.xaxis = xedges if type(xedges) is Axis else Axis(xedges)
        h.yaxis = yedges if type(yedges) is Axis else Axis(yedges)
        h.zaxis = zedges if type(yedges) is Axis else Axis(yedges)
        return h

    def FindBin(self, x, y, z, gid=False):
        ix = self.xaxis.FindBin(x)
        iy = self.yaxis.FindBin(y)
        iz = self.zaxis.FindBin(z)
        if not gid:
            return ix, iy, iz
        else:
            return (ix * self.n.shape[1] * self.n.shape[2]) + iy * self.n.shape[2] + iz

    def ToH5(self, file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'w') as f:
            dset = f.create_dataset(path, data=self.n, compression='gzip')
            dset.attrs['xaxis'] = self.xaxis.edges
            dset.attrs['yaxis'] = self.yaxis.edges
            dset.attrs['zaxis'] = self.zaxis.edges
            
    def ToROOT(self, output, name):
        import ROOT
        from array import array
        h = ROOT.TH2D('', '',
                      len(self.xaxis.edges), array('d', self.xaxis.edges),
                      len(self.yaxis.edges), array('d', self.yaxis.edges))
        
        for i in range(len(self.xaxis.edges)):
            h.SetBinContent(i+1, self.n[i])
        h.Write(name)

    def ToHist1D(self):
        return Hist1D.Filled(self.n.flatten(),
                             np.linspace(0,
                                         np.prod(*self.n.shape),
                                         np.prod(*self.n.shape)+1))
    
    def Draw(self, ax, colorbar=True,**kwargs):
        raise NotImplementedError

        
    


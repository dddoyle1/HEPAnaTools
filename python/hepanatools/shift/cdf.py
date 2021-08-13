import numpy as np
import pandas as pd

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
    def __init__(self, bins):
        self.bins = bins
        
    def find_bin(self, x, **kwargs):
        return np.digitize(x, self.bins, **kwargs)[0]

    def get_bin_center(self, i):
        return (self.bins[i+1] + self.bins[i]) / 2


class Hist:
    def __init__(self, X, bins, **kwargs):
        self.n, edges = np.histogram(X, bins, **kwargs)
        self.xaxis = Axis(edges)

    @staticmethod
    def FromH5(file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'r') as f:
            dset = f.get(path)[:]
            h = Hist.FilledHist(dset[:], dset.attrs['xbins'])
        return h

    @staticmethod
    def Filled(n, bins):
        h = Hist([], [])
        h.n = n
        h.xaxis = bins if type(bins) is Axis else Axis(bins)
        return h
        

    def ToH5(self, file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'w') as f:
            dset = f.create_dataset(path, data=self.n, compression='gzip')
            dset.attrs['xaxis'] = self.xaxis.bins

class Hist2D:
    def __init__(self, X, Y, bins, **kwargs):
        self.n, xedges, yedges = np.histogram(X, Y, bins, **kwargs)
        self.xaxis = Axis(xedges)
        self.yaxis = Axis(yedges)

    @staticmethod
    def FromH5(file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'r') as f:
            dset = f.get(path)[:]
            h = Hist2D.FilledHist(dset[:], dset.attrs['xbins'], dset.attrs['ybins'])
        return h

    @staticmethod
    def Filled(n, xbins, ybins):
        h = Hist([], [], [])
        h.n = n
        h.xaxis = xbins if type(xbins) is Axis else Axis(xbins)
        h.yaxis = ybins if type(ybins) is Axis else Axis(ybins)
        return h
        

    def ToH5(self, file_name_or_handle, path):
        with FileNameOrHandle(file_name_or_handle, 'w') as f:
            dset = f.create_dataset(path, data=self.n, compression='gzip')
            dset.attrs['xaxis'] = self.xaxis.bins
            dset.attrs['yaxis'] = self.yaxis.bins
            

class PDF:
    def __init__(self, X, Y, bins):    
        self._bins = bins
        self._n = np.histogram2d(X, Y,
                                 bins=bins,
                                 density=True)

        


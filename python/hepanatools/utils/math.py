import numpy as np
import numba

@numba.jit(nopython=True)
def chisq(x, u):
    return ((x - u)**2 / u).sum()


    

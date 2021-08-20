import numpy as np
import numba


@numba.jit(nopython=True)
def chisq(x, u):
    return ((x - u) ** 2 / u).sum()


@numba.jit(nopython=True)
def not_quite_chisq(x, u):
    return (((x - u) / u) ** 2).sum()

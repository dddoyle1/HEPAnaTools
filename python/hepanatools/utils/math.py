import numpy as np
import numba


@numba.jit(nopython=True)
def chisq(x, u):
    return ((x - u) ** 2 / u).sum()


@numba.jit(nopython=True)
def not_quite_chisq(x, u):
    return (((x - u) / u) ** 2).sum()


@numba.jit(nopython=True)
def covariance_matrix(nominal, shift):
    return [
        [
            (shift[i] - nominal[i]) * (shift[j] - nominal[j])
            for i in range(nominal.shape[0])
        ]
        for j in range(nominal.shape[0])
    ]


@numba.jit(nopython=True)
def correlation_matrix(nominal, shift):
    return [
        [
            (shift[i] - nominal[i])
            * (shift[j] - nominal[j])
            / np.sqrt(abs(nominal[i] - shift[i]) * abs(nominal[j] - shift[j]))
            for i in range(nominal.shape[0])
        ]
        for j in range(nominal.shape[0])
    ]

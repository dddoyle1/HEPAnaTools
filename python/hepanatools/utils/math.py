import numpy as np
import numba
import scipy


def mv_bin_sigma(vals, nsigma, cv):
    cv_idx = 0
    cv_center = 0
    sorted_vals = np.sort(vals)
    nvals = len(vals)
    for i in range(nvals - 1):
        if sorted_vals[i] <= cv and cv < sorted_vals[i + 1]:
            cv_idx = i
            break
    cv_center = cv_idx + 0.5
    count_fraction = scipy.special.erf(nsigma / np.sqrt(2))
    nsideevents = 0
    lastbinidx = nvals - 1
    if nsigma >= 0:
        nsideevents = lastbinidx - cv_idx
    else:
        nsideevents = cv_idx
    boundidx = cv_center + count_fraction * nsideevents

    idx = 0
    if nsigma >= 0:
        index = min(boundidx, nvals - 1)
    else:
        index = max(boundidx, 0)
    return vals[int(index)]


def mv_cv_and_uncert(universes):
    cv = universes.mean(axis=0)
    up = np.array(
        [mv_bin_sigma(universes[:, i], 1, cv[i]) for i in range(universes.shape[1])]
    )
    dw = np.array(
        [mv_bin_sigma(universes[:, i], -1, cv[i]) for i in range(universes.shape[1])]
    )
    return cv, up, dw


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

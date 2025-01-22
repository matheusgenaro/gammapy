# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
import numpy as np
from decimal import *

cimport numpy as np
cimport cython


cdef extern from "math.h":
    float log(float x)

global TRUNCATION_VALUE
TRUNCATION_VALUE = 1e-25

@cython.cdivision(True)
@cython.boundscheck(False)
def basil_sum_cython(np.ndarray[np.float_t, ndim=1] counts,
                    np.ndarray[np.float_t, ndim=1] npred_s,
                    np.ndarray[np.float_t, ndim=1] npred_b,
                    np.ndarray[np.float_t, ndim=1] comb):
    """Summed BASiL 3D fit statistics.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts array.
    npred_s : `~numpy.ndarray`
        Predicted counts array.
    npred_b : `~numpy.ndarray`
        Predicted background counts array.
    comb : `~numpy.ndarray`
        Combinatory factor array.
    """
    cdef np.float_t sum = 0
    cdef np.float_t npr, lognpr
    cdef unsigned int i, ni, k, j
    cdef np.float_t trunc = TRUNCATION_VALUE
    cdef np.float_t logtrunc = log(TRUNCATION_VALUE)

    ni = counts.shape[0]
    k = 0
    for i in range(ni):
        npr = npred_s[i] + npred_b[i]
        if npr < trunc:
            npr = trunc
        if int(counts[i]) == 0:
            logterm = 0
        else:
            logterm = Decimal(0)
            if (npred_s[i] == 0) & (npred_b[i] != 0):
                logterm += Decimal(comb[i+k])*(Decimal(npred_b[i])**Decimal(int(counts[i])))
            elif (npred_s[i] != 0) & (npred_b[i] == 0):
                logterm += Decimal(comb[i+k+int(counts[i])])*(Decimal(npred_s[i])**Decimal(int(counts[i])))
            else:
                for j in range(int(counts[i])+1):
                    logterm += Decimal(comb[i+k+j])*(Decimal(npred_s[i])**Decimal(j))*(Decimal(npred_b[i])**Decimal(int(counts[i])-j))
                k += int(counts[i])
            if logterm < trunc:
                logterm = logtrunc
            else:
                logterm = float(logterm.log10()) / np.log10(np.exp(1))
        sum += (npr - logterm)

    return 2 * sum

@cython.cdivision(True)
@cython.boundscheck(False)
def cash_sum_cython(np.ndarray[np.float_t, ndim=1] counts,
                    np.ndarray[np.float_t, ndim=1] npred):
    """Summed cash fit statistics.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts array.
    npred : `~numpy.ndarray`
        Predicted counts array.
    """
    cdef np.float_t sum = 0
    cdef np.float_t npr, lognpr
    cdef unsigned int i, ni
    cdef np.float_t trunc = TRUNCATION_VALUE
    cdef np.float_t logtrunc = log(TRUNCATION_VALUE)

    ni = counts.shape[0]
    for i in range(ni):
        npr = npred[i]
        if npr > trunc:
            lognpr = log(npr)
        else:
            npr = trunc
            lognpr = logtrunc

        sum += npr
        if counts[i] > 0:
            sum -= counts[i] * lognpr

    return 2 * sum


@cython.cdivision(True)
@cython.boundscheck(False)
def f_cash_root_cython(np.float_t x, np.ndarray[np.float_t, ndim=1] counts,
                       np.ndarray[np.float_t, ndim=1] background,
                       np.ndarray[np.float_t, ndim=1] model):
    """Function to find root of. Described in Appendix A, Stewart (2009).

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count image slice, where model is defined.
    background : `~numpy.ndarray`
        Background image slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    cdef np.float_t sum = 0
    cdef unsigned int i, ni
    ni = counts.shape[0]
    for i in range(ni):
        if model[i] > 0:
            if counts[i] > 0:
                sum += model[i] * (1 - counts[i] / (x * model[i] + background[i]))
            else:
                sum += model[i]

    # 2 is required to maintain the correct normalization of the
    # derivative of the likelihood function. It doesn't change the result of
    # the fit.
    return 2 * sum


@cython.cdivision(True)
@cython.boundscheck(False)
def norm_bounds_cython(np.ndarray[np.float_t, ndim=1] counts,
                            np.ndarray[np.float_t, ndim=1] background,
                            np.ndarray[np.float_t, ndim=1] model):
    """Compute bounds for the root of `_f_cash_root_cython`.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts image
    background : `~numpy.ndarray`
        Background image
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    cdef np.float_t s_model = 0, s_counts = 0, sn, sn_min = 1e14, c_min = 1
    cdef np.float_t b_min, b_max, sn_min_total = 1e14
    cdef unsigned int i, ni
    ni = counts.shape[0]
    for i in range(ni):
        if counts[i] > 0:
            s_counts += counts[i]
            if model[i] > 0:
                sn = background[i] / model[i]
                if sn < sn_min:
                    sn_min = sn
                    c_min = counts[i]
        if model[i] > 0:
            s_model += model[i]
            sn = background[i] / model[i]
            if sn < sn_min_total:
                sn_min_total = sn
    b_min = c_min / s_model - sn_min
    b_max = s_counts / s_model - sn_min
    return b_min, b_max, -sn_min_total

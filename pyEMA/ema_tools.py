import numpy as np
from tqdm import tqdm

from . import tools


def redundant_values(omega, xi, prec):
    """
    This function supresses the redundant values of frequency and damping
    vectors, which are the consequence of conjugate values

    :param omega: eiqenfrquencies vector
    :param xi: damping ratios vector
    :param prec: absoulute precision in order to distinguish between two values
    """

    N = len(omega)
    test_omega = np.zeros((N, N), dtype='int')
    for i in range(1, N):
        for j in range(0, i):
            if np.abs((omega[i] - omega[j])) < prec:
                test_omega[i, j] = 1
            else:
                test_omega[i, j] = 0

    test = np.sum(test_omega, axis=0)

    omega_mod = omega[np.argwhere(test < 1)]
    xi_mod = xi[np.argwhere(test < 1)]

    return omega_mod, xi_mod


def stabilisation(sr, nmax, err_fn, err_xi):
    """
    A function that computes the stabilisation matrices needed for the
    stabilisation chart. The computation is focused on comparison of
    eigenfrequencies and damping ratios in the present step 
    (N-th model order) with the previous step ((N-1)-th model order). 

    :param sr: list of lists of complex natrual frequencies
    :param n: maximum number of degrees of freedom
    :param err_fn: relative error in frequency
    :param err_xi: relative error in damping

    :return fn_temap eigenfrequencies matrix
    :return xi_temp: updated damping matrix
    :return test_fn: updated eigenfrequencies stabilisation test matrix
    :return test_xi: updated damping stabilisation test matrix
    """

    # TODO: check this later for optimisation # this doffers by LSCE and LSCF
    fn_temp = np.zeros((2*nmax, nmax), dtype='double')
    xi_temp = np.zeros((2*nmax, nmax), dtype='double')
    test_fn = np.zeros((2*nmax, nmax), dtype='int')
    test_xi = np.zeros((2*nmax, nmax), dtype='int')

    for nr, n in enumerate(tqdm(range(nmax), ncols=100)):
        fn, xi = tools.complex_freq_to_freq_and_damp(sr[nr])
        # elimination of conjugate values in
        fn, xi = redundant_values(fn, xi, 1e-3)
        # order to decrease computation time
        if n == 1:
            # first step
            fn_temp[0:len(fn), 0:1] = fn
            xi_temp[0:len(fn), 0:1] = xi

        else:
            # Matrix test is created for comparison between present(N-th) and
            # previous (N-1-th) data (eigenfrequencies). If the value equals:
            # --> 1, the data is within relative tolerance err_fn
            # --> 0, the data is outside the relative tolerance err_fn
            fn_test = np.zeros((len(fn), len(fn_temp[:, n - 1])), dtype='int')
            xi_test = np.zeros((len(xi), len(xi_temp[:, n - 1])), dtype='int')

            for i in range(len(fn)):
                fn_test[i, np.abs((fn[i] - fn_temp[:, n-2]) /
                                  fn_temp[:, n-2]) < err_fn] = 1
                xi_test[i, np.abs((xi[i] - xi_temp[:, n-2]) /
                                  xi_temp[:, n-2]) < err_xi] = 1

                fn_temp[i, n - 1] = fn[i]
                xi_temp[i, n - 1] = xi[i]

                test_fn[i, n-1] = np.sum(fn_test[i, :2*n])
                test_xi[i, n-1] = np.sum(xi_test[i, :2*n])

    return fn_temp, xi_temp, test_fn, test_xi


def irfft_adjusted_lower_limit(x, low_lim, indices):
    """
    Compute the ifft of real matrix x with adjusted summation limits:
    ::
        y(j) = sum[k=-n-2, ... , -low_lim-1, low_lim, low_lim+1, ... n-2, n-1] x[k] * exp(sqrt(-1)*j*k* 2*pi/n),
        j =-n-2, ..., -low_limit-1, low_limit, low_limit+1, ... n-2, n-1

    :param x: Single-sided real array to Fourier transform.
    :param low_lim: lower limit index of the array x.
    :param indices: list of indices of interest
    :return: Fourier transformed two-sided array x with adjusted lower limit.
             Retruns values.

    Source: https://github.com/openmodal/OpenModal/blob/master/OpenModal/fft_tools.py
    """

    nf = 2 * (x.shape[1] - 1)
    a = (np.fft.irfft(x, n=nf)[:, indices]) * nf
    b = (np.fft.irfft(x[:, :low_lim], n=nf)[:, indices]) * nf

    return a - b


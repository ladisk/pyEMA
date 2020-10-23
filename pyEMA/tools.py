import numpy as np
from tqdm import tqdm


def complex_freq_to_freq_and_damp(sr):
    """
    Convert the complex natural frequencies to natural frequencies and the
    corresponding dampings.

    :param sr: complex natural frequencies
    :return: natural frequency and damping
    """
    fr = np.sign(np.imag(sr)) * np.abs(sr)
    xir = -sr.real/fr
    fr /= (2 * np.pi)

    return fr, xir


def MAC(phi_X, phi_A):
    """Modal Assurance Criterion.

    Literature:
        [1] Maia, N. M. M., and J. M. M. Silva. 
            "Modal analysis identification techniques." Philosophical
            Transactions of the Royal Society of London. Series A: 
            Mathematical, Physical and Engineering Sciences 359.1778 
            (2001): 29-40. 

    :param phi_X: Mode shape matrix X, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :param phi_A: Mode shape matrix A, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: MAC matrix. Returns MAC value if both ``phi_X`` and ``phi_A`` are
        one-dimensional arrays.
    """
    if phi_X.ndim == 1:
        phi_X = phi_X[:, np.newaxis]
    
    if phi_A.ndim == 1:
        phi_A = phi_A[:, np.newaxis]
    
    if phi_X.ndim > 2 or phi_A.ndim > 2:
        raise Exception(f'Mode shape matrices must have 1 or 2 dimensions (phi_X: {phi_X.ndim}, phi_A: {phi_A.ndim})')

    if phi_X.shape[0] != phi_A.shape[0]:
        raise Exception(f'Mode shapes must have the same first dimension (phi_X: {phi_X.shape[0]}, phi_A: {phi_A.shape[0]})')

    MAC = np.abs(np.conj(phi_X).T @ phi_A)**2
    for i in range(phi_X.shape[1]):
        for j in range(phi_A.shape[1]):
            MAC[i, j] = MAC[i, j]/\
                            (np.conj(phi_X[:, i]) @ phi_X[:, i] *\
                            np.conj(phi_A[:, j]) @ phi_A[:, j])

    
    if MAC.shape == (1, 1):
        MAC = MAC[0, 0]

    return MAC


def MSF(phi_X, phi_A):
    """Modal Scale Factor.

    If ``phi_X`` and ``phi_A`` are matrices, multiple msf are returned.

    Scales ``phi_X`` to ``phi_A`` when multiplying: ``msf*phi_X``. 
    Also takes care of 180 deg phase difference.

    :param phi_X: Mode shape matrix X, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :param phi_A: Mode shape matrix A, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: np.ndarray, MSF values
    """
    if phi_X.ndim == 1:
        phi_X = phi_X[:, None]
    if phi_A.ndim == 1:
        phi_A = phi_A[:, None]
    
    if phi_X.shape[0] != phi_A.shape[0] or phi_X.shape[1] != phi_A.shape[1]:
        raise Exception(f'`phi_X` and `phi_A` must have the same shape: {phi_X.shape} and {phi_A.shape}')

    n_modes = phi_X.shape[1]
    msf = []
    for i in range(n_modes):
        _msf = (phi_A[:, i].T @ phi_X[:, i]) / \
                (phi_X[:, i].T @ phi_X[:, i])

        msf.append(_msf)

    return np.array(msf).real

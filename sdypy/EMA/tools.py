import numpy as np
from tqdm import tqdm


def complex_freq_to_freq_and_damp(sr):
    """
    Convert the complex natural frequencies to natural frequencies and the
    corresponding dampings.

    A complex natural frequency is defined:
    
    .. math::

        \\lambda_r = -\\zeta\\,\\omega_r \\pm \\mathrm{i}\\,\\omega_r\\,\\sqrt{1-\\zeta^2},
    
    where :math:`\\lambda_r` is the :math:`r` th complex natural frequency and :math:`\\omega_r`
    and :math:`\\zeta_r` are the :math:`r` th natural frequency [rad/s] and damping, respectively.

    :param sr: complex natural frequencies
    :return: natural frequency [Hz] and damping
    """
    # Extract natural frequency
    fr = np.sign(np.imag(sr)) * np.abs(sr)
    
    # Extract damping
    xir = -sr.real/fr
    
    # Convert natural frequency to Hz
    fr /= (2 * np.pi)

    return fr, xir


def MAC(phi_X, phi_A):
    """Modal Assurance Criterion.

    The number of locations (axis 0) must be the same for ``phi_X`` and
    ``phi_A``. The nubmer of modes (axis 1) is arbitrary.

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

    The MAF scales ``phi_X`` to ``phi_A`` when multiplying: ``msf*phi_X``. 
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


def MCF(phi):
    """ Modal complexity factor.

    The MCF ranges from 0 to 1. It returns 0 for real modes and 1 for complex modes. 
    When ``dtype`` of ``phi`` is ``complex``, the modes can still be real, if the angles 
    of all components are the same.

    Additional information on MCF:
    http://www.svibs.com/resources/ARTeMIS_Modal_Help/Generic%20Complexity%20Plot.html
    
    :param phi: Complex mode shape matrix, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: MCF (a value between 0 and 1)
    """
    if phi.ndim == 1:
        phi = phi[:, None]
    n_modes = phi.shape[1]
    mcf = []
    for i in range(n_modes):
        S_xx = np.dot(phi[:, i].real, phi[:, i].real)
        S_yy = np.dot(phi[:, i].imag, phi[:, i].imag)
        S_xy = np.dot(phi[:, i].real, phi[:, i].imag)
        
        _mcf = 1 - ((S_xx - S_yy)**2 + 4*S_xy**2) / (S_xx + S_yy)**2
        
        mcf.append(_mcf)
    return np.array(mcf)

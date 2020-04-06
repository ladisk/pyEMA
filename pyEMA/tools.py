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
    """
    Modal Assurance Criterion.

    Literature:
        [1] Maia, N. M. M., and J. M. M. Silva. 
            "Modal analysis identification techniques." Philosophical
            Transactions of the Royal Society of London. Series A: 
            Mathematical, Physical and Engineering Sciences 359.1778 
            (2001): 29-40. 

    :param phi_X: Mode shape matrix X
    :param phi_A: Mode shape matrix A
    :return: MAC matrix
    """
    if phi_X.shape != phi_A.shape:
        raise Exception('Mode shape matrices must be of the same dimension.')
    modes = phi_X.shape[1]
    MAC = np.abs(np.conj(phi_X).T @ phi_A)**2
    for i in range(modes):
        for j in range(modes):
            MAC[i, j] = MAC[i, j]/\
                            (np.conj(phi_X[:, i]) @ phi_X[:, i] *\
                            np.conj(phi_A[:, j]) @ phi_A[:, j])
    return MAC


def complex_to_normal_mode(mode):
    """Transform a complex mode shape to normal mode shape.
    
    The real mode shape should have the maximum correlation with
    the original complex mode shape. The vector that is most correlated
    with the complex mode, is the real part of the complex mode when it is
    rotated so that the norm of its real part is maximized. [1]
    
    Literature:
        [1] Gladwell, H. Ahmadian GML, and F. Ismail. 
            "Extracting Real Modes from Complex Measured Modes."
    
    :params mode: np.ndarray, a mode shape to be transformed. Can contain a single
        mode shape or a modal matrix `(n_locations, n_modes)`.
    :return: normal mode shape
    """
    if mode.ndim == 1:
        mode = mode[None, :, None]
    elif mode.ndim == 2:
        mode = mode.T[:, :, None]
    else:
        raise Exception(f'`mode` must have 1 or 2 dimensions ({mode.ndim}).')

    # Normalize modes so that norm == 1.0
    mode = np.array([m/np.linalg.norm(m) for m in mode])
    
    mode_T = np.transpose(mode, [0, 2, 1])

    U = np.real(mode) @ np.real(mode_T) + np.imag(mode) @ np.imag(mode_T)
    
    val, vec = np.linalg.eig(U)
    i = np.argmax(np.real(val), axis=1)

    normal_mode = np.real([v[:, _] for v, _ in zip(vec, i)]).T

    return normal_mode
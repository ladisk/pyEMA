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

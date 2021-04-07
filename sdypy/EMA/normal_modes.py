import numpy as np

def complex_to_normal_mode(mode, max_dof=50, long=True):
    """Transform a complex mode shape to normal mode shape.
    
    The real mode shape should have the maximum correlation with
    the original complex mode shape. The vector that is most correlated
    with the complex mode, is the real part of the complex mode when it is
    rotated so that the norm of its real part is maximized. [1]

    ``max_dof`` and ``long`` arguments are given for modes that have
    a large number of degrees of freedom. See ``_large_normal_mode_approx()``
    for more details.
    
    Literature:
        [1] Gladwell, H. Ahmadian GML, and F. Ismail. 
            "Extracting Real Modes from Complex Measured Modes."
    
    :param mode: np.ndarray, a mode shape to be transformed. Can contain a single
        mode shape or a modal matrix `(n_locations, n_modes)`.
    :param max_dof: int, maximum number of degrees of freedom that can be in
        a mode shape. If larger, ``_large_normal_mode_approx()`` function
        is called. Defaults to 50.
    :param long: bool, If True, the start in stepping itartion is altered, the
        angles of rotation are averaged (more in ``_large_normal_mode_approx()``).
        This is needed only when ``max_dof`` is exceeded. The normal modes are 
        more closely related to the ones computed with an entire matrix. Defaults to True.
    :return: normal mode-shape
    """
    if mode.ndim == 1:
        mode = mode[None, :, None]
    elif mode.ndim == 2:
        mode = mode.T[:, :, None]
    else:
        raise Exception(f'`mode` must have 1 or 2 dimensions ({mode.ndim}).')
    
    if mode.shape[1] > max_dof:
        return _large_normal_mode_approx(mode[:, :, 0].T, step=int(np.ceil(mode.shape[1] / max_dof)) + 1, long=long)
    
    # Normalize modes so that norm == 1.0
    _norm = np.linalg.norm(mode, axis=1)[:, None, :]
    mode = mode / _norm

    mode_T = np.transpose(mode, [0, 2, 1])

    U = np.matmul(np.real(mode), np.real(mode_T)) + np.matmul(np.imag(mode), np.imag(mode_T))

    val, vec = np.linalg.eig(U)
    i = np.argmax(np.real(val), axis=1)

    normal_mode = np.real([v[:, _] for v, _ in zip(vec, i)]).T
    return normal_mode


def _large_normal_mode_approx(mode, step, long):
    """Get normal mode approximation for large modes.
    
    In cases, where ``mode`` has ``n`` coordinates and
    ``n`` is large, this would result in a matrix ``U`` of
    size ``n x n``. To find eigenvalues of this non-sparse
    matrix is computationally expensive. The solution is to
    find the angle of the rotation for the vector - this is
    done using only every ``step`` element of ``mode``.
    The entire ``mode`` is then rotated, thus the full normal
    mode is obtained.
    
    To ensure the influence of all the coordinates, a ``long``
    parameter can be used. Multiple angles of rotation are
    computed and then averaged.
    
    :param mode: a 2D mode shape or modal matrix ``(n_locations x n_modes)``
    :param step: int, every ``step`` elemenf of ``mode`` will be taken
        into account for angle of rotation calculation.
    :param long: bool, if True, the angle of rotation is computed
        iteratively for different starting positions (from 0 to ``step``), when
        every ``step`` element is taken into account.
    :return: normal mode or modal matrix of ``mode``.
    """
    if mode.ndim == 1:
        mode = mode[:, None]
    elif mode.ndim > 2:
        raise Exception(f'`mode` must have 1 or 2 dimensions ({mode.ndim})')
        
    mode = mode / np.linalg.norm(mode, axis=0)[None, :]
    
    if long:
        step_long = step
    else:
        step_long = 1
    
    Alpha = []
    for i in range(step_long):
        mode_step = mode[i::step]
        mode_normal_step = complex_to_normal_mode(mode_step)

        v1 = np.concatenate((np.real(mode_step)[:, :, None], np.imag(mode_step)[:, :, None]), axis=2)
        v2 = np.concatenate((np.real(mode_normal_step)[:, :, None], np.imag(mode_normal_step)[:, :, None]), axis=2)
        
        v1 /= np.linalg.norm(v1, axis=2)[:, :, None]
        v2 /= np.linalg.norm(v2, axis=2)[:, :, None]
        
        dot_product = np.array([np.matmul(np.transpose(v1[:, j, :, None], [0, 2, 1]), v2[:, j, :, None]) for j in range(v1.shape[1])])
        angles = np.arccos(dot_product)

        alpha = np.mean(angles[:, :, 0, 0], axis=1)
        Alpha.append(alpha)

    alpha = np.mean(Alpha, axis=0)[None, :]

    mode_normal_full = np.real(mode)*np.cos(alpha) - np.imag(mode)*np.sin(alpha)
    mode_normal_full /= np.linalg.norm(mode_normal_full, axis=0)[None, :]
    
    return mode_normal_full


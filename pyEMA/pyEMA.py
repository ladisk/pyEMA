import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg
from tqdm import tqdm
from scipy.linalg import toeplitz, companion
from scipy.optimize import least_squares, leastsq

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from .pole_picking import SelectPoles
from . import tools
from . import stabilization
from . import normal_modes

class Model():
    """
    Modal model of frequency response functions.
    """

    def __init__(self,
                 frf=None,
                 freq=None,
                 dt=None,
                 lower=50,
                 upper=10000,
                 pol_order_high=100,
                 pyfrf=False,
                 get_partfactors=False):
        """
        :param frf: Frequency response function matrix (must be receptance!)
        :type frf: ndarray
        :param freq: Frequency array
        :type freq: array
        :param lower: Lower limit for pole determination [Hz]
        :type lower: int, float
        :param upper: Upper limit for pole determination [Hz]
        :type upper: int, float
        :param pol_order_high: Highest order of the polynomial
        :type pol_order_high: int
        """
        try:
            self.lower = float(lower)
        except:
            raise Exception('lower must be float or integer')
        if self.lower < 0:
            raise Exception('lower must be positive or equal to zero')

        try:
            self.upper = float(upper)
        except:
            raise Exception('upper must be flaot or integer')
        if self.upper < self.lower:
            raise Exception('upper must be greater than lower')

        if pyfrf:
            self.frf = 0
        elif not pyfrf and frf is not None and freq is not None:
            try:
                self.frf = np.asarray(frf)
            except:
                raise Exception('cannot contert frf to numpy ndarray')
            if self.frf.ndim == 1:
                self.frf = np.array([self.frf])

            try:
                self.freq = np.asarray(freq)
            except:
                raise Exception('cannot convert freq to numpy array')
            if self.freq.ndim != 1:
                raise Exception(
                    f'ndim of freq is not equal to 1 ({self.freq.ndim})')

            # Cut off the frequencies above 'upper' argument
            cutoff_ind = np.argmin(np.abs(self.freq - self.upper))
            self.frf = self.frf[:, :cutoff_ind]
            self.freq = self.freq[:cutoff_ind]
        else:
            raise Exception('input arguments are not defined')

        try:
            self.pol_order_high = int(pol_order_high)
        except:
            raise Exception('cannot convert pol_order_high to integer')
        if self.pol_order_high <= 0:
            raise Exception('pol_order_high must be positive')

        if not pyfrf:
            self.omega = 2 * np.pi * self.freq
            if dt is None:
                self.sampling_time = 1/(2*self.freq[-1])
            else:
                self.sampling_time = dt
        
        self.get_participation_factors = get_partfactors

    def add_frf(self, pyfrf_object):
        """
        Add a FRF at a next location.

        This method can be used in relation to pyFRF from Open Modal (https://github.com/openmodal)

        :param pyfrf_object: FRF object from pyFRF
        :type pyfrf_object: object
        """
        freq = pyfrf_object.get_f_axis()
        sel = (freq >= 1.0e-1)

        self.freq = freq[sel]
        self.omega = 2 * np.pi * self.freq
        self.sampling_time = 1/(2*self.freq[-1])

        new_frf = np.vstack(pyfrf_object.get_FRF(form='receptance')[sel])

        if isinstance(self.frf, int):
            self.frf = new_frf.T
        else:
            self.frf = np.concatenate((self.frf, new_frf.T), axis=0)

    def get_poles(self, method='lscf', show_progress=True):
        """Compute poles based on polynomial approximation of FRF.

        Source: https://github.com/openmodal/OpenModal/blob/master/OpenModal/analysis/lscf.py

        The LSCF method is an frequency-domain Linear Least Squares
        estimator optimized  for modal parameter estimation. The choice of
        the most important algorithm characteristics is based on the
        results in [1] (Section 5.3.3.) and can be summarized as:

            - Formulation: the normal equations [1]
            (Eq. 5.26: [sum(Tk - Sk.H * Rk^-1 * Sk)]*ThetaA=D*ThetaA = 0)
            are constructed for the common denominator discrete-time
            model in the Z-domain. Consequently, by looping over the
            outputs and inputs, the submatrices Rk, Sk, and Tk are
            formulated through the use of the FFT algorithm as Toeplitz
            structured (n+1) square matrices. Using complex coefficients,
            the FRF data within the frequency band of interest (FRF-zoom)
            is projected in the Z-domain in the interval of [0, 2*pi] in
            order to improve numerical conditioning. (In the case that
            real coefficients are used, the data is projected in the
            interval of [0, pi].) The projecting on an interval that does
            not completely describe the unity circle, say [0, alpha*2*pi]
            where alpha is typically 0.9-0.95. Deliberately over-modeling
            is best applied to cope with discontinuities. This is
            justified by the use of a discrete time model in the Z-domain,
            which is much more robust for a high order of the transfer
            function polynomials.

            - Solver: the normal equations can be solved for the
            denominator coefficients ThetaA by computing the Least-Squares
            (LS) or mixed Total-Least-Squares (TLS) solution. The inverse
            of the square matrix D for the LS solution is computed by
            means of a pseudo inverse operation for reasons of numerical
            stability, while the mixed LS-TLS solution is computed using
            an SVD (Singular Value Decomposition).

        Literature:
            [1] Verboven, P., Frequency-domain System Identification for
                Modal Analysis, Ph. D. thesis, Mechanical Engineering Dept.
                (WERK), Vrije Universiteit Brussel, Brussel, (Belgium),
                May 2002, (http://mech.vub.ac.be/avrg/PhD/thesis_PV_web.pdf)
            [2] Verboven, P., Guillaume, P., Cauberghe, B., Parloo, E. and
                Vanlanduit S., Stabilization Charts and Uncertainty Bounds
                For Frequency-Domain Linear Least Squares Estimators, Vrije
                Universiteit Brussel(VUB), Mechanical Engineering Dept.
                (WERK), Acoustic and Vibration Research Group (AVRG),
                Pleinlaan 2, B-1050 Brussels, Belgium,
                e-mail: Peter.Verboven@vub.ac.be, url:
                (http://sem-proceedings.com/21i/sem.org-IMAC-XXI-Conf-s02p01
                -Stabilization-Charts-Uncertainty-Bounds-Frequency-Domain-
                Linear-Least.pdf)
            [3] P. Guillaume, P. Verboven, S. Vanlanduit, H. Van der
                Auweraer, B. Peeters, A Poly-Reference Implementation of the
                Least-Squares Complex Frequency-Domain Estimator, Vrije
                Universiteit Brussel, LMS International

        :param method: The method of poles calculation.
        :param show_progress: Show progress bar
        """
        if method != 'lscf':
            raise Exception(
                f'no method "{method}". Currently only "lscf" method is implemented.')

        if show_progress:
            def tqdm_range(x): return tqdm(x, ncols=100)
        else:
            def tqdm_range(x): return x

        self.all_poles = []
        self.pole_freq = []
        self.pole_xi = []
        self.partfactors = []

        lower_ind = np.argmin(np.abs(self.freq - self.lower))
        n = self.pol_order_high * 2
        nf = 2 * (self.frf.shape[1] - 1)
        nr = self.frf.shape[0]

        indices_s = np.arange(-n, n+1)
        indices_t = np.arange(n+1)

        sk = -_irfft_adjusted_lower_limit(self.frf, lower_ind, indices_s)
        t = _irfft_adjusted_lower_limit(
            self.frf.real**2 + self.frf.imag**2, lower_ind, indices_t)
        r = -(np.fft.irfft(np.ones(lower_ind), n=nf))[indices_t]*nf
        r[0] += nf

        s = []
        for i in range(nr):
            s.append(toeplitz(sk[i, n:], sk[i, :n+1][::-1]))
        t = toeplitz(np.sum(t[:, :n+1], axis=0))
        r = toeplitz(r)

        # Ascending polinomial order pole computation
        for j in tqdm_range(range(2, n+1, 2)):
            d = 0
            rinv = np.linalg.inv(r[:j+1, :j+1])
            for i in range(nr):
                snew = s[i][:j+1, :j+1]
                d -= np.dot(np.dot(snew[:j+1, :j+1].T,
                                   rinv), snew[:j+1, :j+1])   # sum
            d += t[:j+1, :j+1]

            a0an1 = np.linalg.solve(-d[0:j, 0:j], d[0:j, j])
            # the numerator coefficients
            sr = np.roots(np.append(a0an1, 1)[::-1])

            # Z-domain (for discrete-time domain model)
            poles = -np.log(sr) / self.sampling_time

            if self.get_participation_factors:
                _t = companion(np.append(a0an1, 1)[::-1])
                _v, _w = np.linalg.eig(_t)
                self.partfactors.append(_w[-1, :])

            f_pole, ceta = tools.complex_freq_to_freq_and_damp(poles)

            self.all_poles.append(poles)
            self.pole_freq.append(f_pole)
            self.pole_xi.append(ceta)

    def select_poles(self):
        _ = SelectPoles(self)

    def stab_chart(self, poles='all', fn_temp=0.001, xi_temp=0.05, legend=True, latex_render=False, title=None):
        """
        Render stability chart.

        Interactive pole selection is possible. Identification of natural 
        frequency and damping coefficients is executed on-the-fly,
        as well as computing reconstructed FRF and modal constants.

        :param poles: poles to be shown on the plot. If 'all', all the poles are shown.
        :param fn_temp: Natural frequency stability crieterion.
        :param xi_temp: Damping stability criterion.
        :param legen: wheather to show the legend
        :param latex_render: if True, the latex render is used for fonts.
        :param title: if given, the stabilization chart is saved under that title.

        The identification can be done in two ways:
        ::
            # 1. Using stability chart
            >>> a.stab_chart() # pick poles
            >>> a.nat_freq # natural frequencies
            >>> a.nat_xi # damping coefficients
            >>> a.H # reconstructed FRF matrix
            >>> a.A # modal constants (a.A[:, -2:] are Lower and Upper residual)

            # 2. Using approximate natural frequencies
            >>> approx_nat_freq = [234, 545]
            >>> a.select_closest_poles(approx_nat_freq)
            >>> a.nat_freq # natural frequencies
            >>> a.nat_xi # damping coefficients
            >>> H, A = a.get_constants(whose_poles='own', FRF_ind='all) # reconstruction
        """
        if poles == 'all':
            poles = self.all_poles

        def replot(init=False):
            """Replot the measured and reconstructed FRF based on new selected poles."""
            ax2.clear()
            ax2.semilogy(self.freq, np.average(
                np.abs(self.frf), axis=0), alpha=0.7, color='k')

            if not init:
                self.H, self.A = self.get_constants(whose_poles='own', FRF_ind='all', least_squares_type='new')
                ax2.semilogy(self.freq, np.average(
                    np.abs(self.H), axis=0), color='r', lw=2)

            ax1.set_xlim([self.lower, self.upper])
            ax1.set_ylim([0, self.pol_order_high+5])

        Nmax = self.pol_order_high
        fn_temp, xi_temp, test_fn, test_xi = stabilization._stabilization(
            poles, Nmax, err_fn=fn_temp, err_xi=xi_temp)

        root = tk.Tk()  # Tkinter
        root.title('Stability Chart')  # Tkinter
        fig = Figure(figsize=(20, 8))  # Tkinter
        ax2 = fig.add_subplot(111)  # Tkinter

        ax1 = ax2.twinx()
        ax1.grid(True)
        replot(init=True)

        ax1.set_xlabel(r'$f$ [Hz]', fontsize=12)
        ax1.set_ylabel(r'Polynomial order', fontsize=12)
        ax2.set_ylabel(r'$|\alpha|$', fontsize=12)

        if latex_render is True:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            ax1.tick_params(axis='both', which='major', labelsize=12)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            ax1.set_xlabel(r'$f$ [Hz]', fontsize=12)
            ax1.set_ylabel(r'Polynomial order', fontsize=12)
            ax2.set_ylabel(r'$|\alpha|_{log}$', fontsize=12)
            ax1.set_xlim([self.lower, self.upper])

        # stable eigenfrequencues, unstable damping ratios
        a = np.argwhere((test_fn > 0) & ((test_xi == 0) | (xi_temp <= 0)))
        # stable eigenfrequencies, stable damping ratios
        b = np.argwhere((test_fn > 0) & ((test_xi > 0) & (xi_temp > 0)))
        # unstable eigenfrequencues, unstable damping ratios
        c = np.argwhere((test_fn == 0) & ((test_xi == 0) | (xi_temp <= 0)))
        # unstable eigenfrequencues, stable damping ratios
        d = np.argwhere((test_fn == 0) & ((test_xi > 0) & (xi_temp > 0)))

        p1 = ax1.plot(fn_temp[a[:, 0], a[:, 1]], 1+a[:, 1], 'bx',
                      markersize=4, label="stable frequency, unstable damping")
        p2 = ax1.plot(fn_temp[b[:, 0], b[:, 1]], 1+b[:, 1], 'gx',
                      markersize=7, label="stable frequency, stable damping")
        p3 = ax1.plot(fn_temp[c[:, 0], c[:, 1]], 1+c[:, 1], 'r.',
                      markersize=4, label="unstable frequency, unstable damping")
        p4 = ax1.plot(fn_temp[d[:, 0], d[:, 1]], 1+d[:, 1], 'r*',
                      markersize=4, label="unstable frequency, stable damping")

        if legend:
            ax1.legend(loc='upper center', ncol=2, frameon=True)
        plt.tight_layout()

        print('SHIFT + LEFT mouse button to pick a pole.\nSHIFT + RIGHT mouse button to erase the last pick.')
        self.nat_freq = []
        self.nat_xi = []
        self.pole_ind = []

        line, = ax1.plot(self.nat_freq, np.repeat(
            self.pol_order_high, len(self.nat_freq)), 'kv', markersize=8)

        # Mark selected poles
        selected, = ax1.plot([], [], 'ko')

        self.shift_is_held = False

        def on_key_press(event):
            """Function triggered on key press (shift)."""
            if event.key == 'shift':
                self.shift_is_held = True

        def on_key_release(event):
            """Function triggered on key release (shift)."""
            if event.key == 'shift':
                self.shift_is_held = False

        def onclick(event):
            # on button 1 press (left mouse button) + shift is held
            if event.button == 1 and self.shift_is_held:
                self.y_data_pole = [event.ydata]
                self.x_data_pole = event.xdata
                self._select_closest_poles_on_the_fly()

                replot()

                print(
                    f'{len(self.nat_freq)}. Frequency: ~{int(np.round(event.xdata))} -->\t{self.nat_freq[-1]} Hz\t(xi = {self.nat_xi[-1]:.4f})')

            # On button 3 press (left mouse button)
            elif event.button == 3 and self.shift_is_held:
                try:
                    del self.nat_freq[-1]  # delete last point
                    del self.nat_xi[-1]
                    del self.pole_ind[-1]
                    replot()
                    print('Deleting the last pick...')
                except:
                    pass

            line.set_xdata(np.asarray(self.nat_freq))  # update data
            line.set_ydata(np.repeat(Nmax*1.04, len(self.nat_freq)))

            selected.set_xdata([self.pole_freq[p[0]][p[1]]
                                for p in self.pole_ind])  # update data
            selected.set_ydata([p[0] for p in self.pole_ind])
            fig.canvas.draw()

        canvas = FigureCanvasTkAgg(fig, root)  # Tkinter
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)  # Tkinter
        NavigationToolbar2Tk(canvas, root)  # Tkinter

        def on_closing():
            if title is not None:
                fig.savefig(title)
            root.destroy()

        # Connecting functions to event manager
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('key_release_event', on_key_release)
        fig.canvas.mpl_connect('button_press_event', onclick)

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()  # Tkinter

    def _select_closest_poles_on_the_fly(self):
        """
        On-the-fly selection of the closest poles.        
        """
        y_ind = int(np.argmin(np.abs(np.arange(0, len(self.pole_freq)
                                               )-self.y_data_pole)))  # Find closest pole order
        # Find cloeset frequency
        sel = np.argmin(np.abs(self.pole_freq[y_ind] - self.x_data_pole))

        self.pole_ind.append([y_ind, sel])
        self.nat_freq.append(self.pole_freq[y_ind][sel])
        self.nat_xi.append(self.pole_xi[y_ind][sel])

    def select_closest_poles(self, approx_nat_freq, f_window=50, fn_temp=0.001, xi_temp=0.05):
        """
        Identification of natural frequency and damping.

        If `approx_nat_freq` is used, the method finds closest poles of the polynomial.

        :param approx_nat_freq: Approximate natural frequency value
        :type approx_nat_freq: list
        :param f_window: width of the optimization frequency window when searching for stable poles
        :type f_window: float, int
        """
        pole_ind = []
        sel_ind = []

        Nmax = self.pol_order_high
        poles = self.all_poles
        fn_temp, xi_temp, test_fn, test_xi = stabilization._stabilization(
            poles, Nmax, err_fn=fn_temp, err_xi=xi_temp)
        # select the stable poles
        b = np.argwhere((test_fn > 0) & ((test_xi > 0) & (xi_temp > 0)))

        mask = np.zeros_like(fn_temp)
        mask[b[:, 0], b[:, 1]] = 1  # mask the unstable poles
        f_stable = fn_temp * mask
        xi_stable = xi_temp * mask
        f_stable[f_stable != f_stable] = 0
        xi_stable[xi_stable != xi_stable] = 0

        self.f_stable = f_stable
        f_windows = [
            f_window//i for i in range(2, 100) if f_window//i > 3] + [2]
        for i, fr in enumerate(approx_nat_freq):
            # Optimize the approximate frequency
            def fun(x, f_step):
                f = x[0]
                _f_stable = f_stable[(f_stable > (fr - f_step))
                                     & (f_stable < (fr + f_step))]
                return _f_stable.flatten() - f

            for f_w in f_windows:
                sol = least_squares(lambda x: fun(x, f_w), x0=[fr])
                fr = sol.x[0]

            # Select the closest frequency
            f_sel = np.argmin(np.abs(f_stable - fr))
            f_sel = np.unravel_index(f_sel, f_stable.shape)

            # The pole index is known (f_sel[1])
            # The frequency index for this pole order is not known
            # A reconstructed pole is compared with existing poles to
            # get the index of the pole.
            sel = np.argmin(np.abs(self.pole_freq[f_sel[1]] - fr))
            selected_pole = -xi_temp[f_sel]*(2*np.pi*fn_temp[f_sel]) + 1j*(
                2*np.pi*fn_temp[f_sel])*np.sqrt(1-xi_temp[f_sel]**2)
            _sel = np.argmin(np.abs(self.all_poles[f_sel[1]] - selected_pole))

            pole_ind.append([f_sel[1], _sel])
            sel_ind.append([f_sel[1], f_sel[0]])

        sel_ind = np.asarray(sel_ind, dtype=int)
        self.pole_ind = np.asarray(pole_ind, dtype=int)

        self.nat_freq = f_stable[sel_ind[:, 1], sel_ind[:, 0]]
        self.nat_xi = xi_stable[sel_ind[:, 1], sel_ind[:, 0]]

    def get_constants(self, method='lsfd', whose_poles='own', FRF_ind='all',
                      f_lower=None, f_upper=None, complex_mode=True, upper_r=True, lower_r=True, least_squares_type='new'):
        """
        Least square frequency domain 1D (Participation factor excluded)

        :param whose_poles: Whose poles to use, defaults to 'own'
        :type whose_poles: object or string ('own'), optional
        :param FRF_ind: FRF at which location to reconstruct, defaults to 'all'
        :type FRF_ind: int or 'all', optional
        :param f_lower: lower limit on frequency for reconstruction. If None, self.lower is used, defaults to None
        :type f_lower: float, optional
        :param f_upper: upper limit on frequency for reconstruction. If None, self.lower is used, defaults to None
        :type f_upper: float, optional
        :param complex_mode: Return complex modes, defaults to True
        :type complex_mode: bool, optional
        :param upper_r: Compute upper residual, defaults to True
        :type upper_r: bool, optional
        :param lower_r: Compute lower residual, defaults to True
        :type lower_r: bool, optional
        :return: modal constants if ``FRF_ind=None``, otherwise reconstructed FRFs and modal constants
        """
        if method != 'lsfd':
            raise Exception(
                f'no method "{method}". Currently only "lsfd" method is implemented.')

        if whose_poles == 'own':
            whose_poles = self

        pole_ind = np.asarray(whose_poles.pole_ind, dtype=int)
        n_poles = pole_ind.shape[0]
        poles = []
        for i in range(n_poles):
            poles.append(whose_poles.all_poles[pole_ind[i, 0]][pole_ind[i, 1]])
        poles = np.asarray(poles)

        # concatenate frequency and FRF array
        if f_lower == None:
            f_lower = self.lower

        if f_upper == None:
            f_upper = self.upper

        lower_ind = np.argmin(np.abs(self.freq - f_lower))
        upper_ind = np.argmin(np.abs(self.freq - f_upper))

        _freq = self.freq[lower_ind:upper_ind]
        _FRF_mat = self.frf[:, lower_ind:upper_ind]
        ome = 2 * np.pi * _freq
        M_2 = len(poles)
        
        def TA_construction(TA_omega):
            len_ome = len(TA_omega)
            if TA_omega[0] == 0: 
                TA_omega[0] = 1.e-2

            _ome = TA_omega[:, np.newaxis]

            # Initialization
            TA = np.zeros([2*len_ome, 2*M_2 + 4])

            # Eigenmodes contribution
            TA[:len_ome, 0:2*M_2:2] =    (-np.real(poles))/(np.real(poles)**2+(_ome-np.imag(poles))**2)+\
                                        (-np.real(poles))/(np.real(poles)**2+(_ome+np.imag(poles))**2)
            TA[len_ome:, 0:2*M_2:2] =    (-(_ome-np.imag(poles)))/(np.real(poles)**2+(_ome-np.imag(poles))**2)+\
                                        (-(_ome+np.imag(poles)))/(np.real(poles)**2+(_ome+np.imag(poles))**2)
            TA[:len_ome, 1:2*M_2+1:2] =  ((_ome-np.imag(poles)))/(np.real(poles)**2+(_ome-np.imag(poles))**2)+\
                                        (-(_ome+np.imag(poles)))/(np.real(poles)**2+(_ome+np.imag(poles))**2)
            TA[len_ome:, 1:2*M_2+1:2] =  (-np.real(poles))/(np.real(poles)**2+(_ome-np.imag(poles))**2)+\
                                        (np.real(poles))/(np.real(poles)**2+(_ome+np.imag(poles))**2)
            
            # Lower and upper residuals contribution
            TA[:len_ome, -4] = -1/(TA_omega**2)
            TA[len_ome:, -3] = -1/(TA_omega**2)
            TA[:len_ome, -2] = np.ones(len_ome)
            TA[len_ome:, -1] = np.ones(len_ome)

            return TA

        AT = np.linalg.pinv(TA_construction(ome))
        FRF_r_i = np.concatenate([np.real(_FRF_mat.T),np.imag(_FRF_mat.T)])
        A_LSFD = AT @ FRF_r_i      
        
        self.A = (A_LSFD[0:2*M_2:2, :] + 1.j*A_LSFD[1:2*M_2+1:2, :]).T
        self.LR = A_LSFD[-4, :]+1.j*A_LSFD[-3, :]
        self.UR = A_LSFD[-2, :]+1.j*A_LSFD[-1, :]
        self.poles = poles

        # FRF reconstruction
        if FRF_ind is None:
            return self.A

        elif FRF_ind == 'all':
            _FRF_r_i = TA_construction(self.omega)@A_LSFD
            frf_ = (_FRF_r_i[:len(self.omega),:] + _FRF_r_i[len(self.omega):,:]*1.j).T
            self.H = frf_
            return frf_, self.A

        elif isinstance(FRF_ind, int):
            frf_ = self.FRF_reconstruct(FRF_ind)[None, :]
            self.H = frf_
            return frf_, self.A

        else:
            raise Exception('FRF_ind must be None, "all" or int')

    def FRF_reconstruct(self, FRF_ind):
        """
        Reconstruct FRF based on modal constants.

        :param FRF_ind: Reconstruct FRF on location with this index, int
        :return: Reconstructed FRF
        """

        FRF_true = np.zeros(len(self.omega), complex)
        for n in range(self.A.shape[1]):
            FRF_true += (self.A[FRF_ind, n] /
                         (1j*self.omega - self.poles[n])) + \
                (np.conjugate(self.A[FRF_ind, n]) /
                 (1j*self.omega - np.conjugate(self.poles[n])))

        FRF_true += -self.LR[FRF_ind] / \
            (self.omega**2) + self.UR[FRF_ind]
        return FRF_true

    def autoMAC(self):
        """
        Auto Modal Assurance Criterion.

        :return: autoMAC matrix
        """
        if not hasattr(self, 'A'):
            raise Exception('Mode shape matrix not defined.')
        return tools.MAC(self.A, self.A)

    def normal_mode(self):
        """Transform the complex mode shape matrix self.A to normal mode shape.
        
        The real mode shape should have the maximum correlation with
        the original complex mode shape. The vector that is most correlated
        with the complex mode, is the real part of the complex mode when it is
        rotated so that the norm of its real part is maximized. [1]
        
        Literature:
            [1] Gladwell, H. Ahmadian GML, and F. Ismail. 
                "Extracting Real Modes from Complex Measured Modes."
        
        :return: normal mode shape
        """
        if not hasattr(self, 'A'):
            raise Exception('Mode shape matrix not defined.')
        
        return normal_modes.complex_to_normal_mode(self.A)

    def print_modal_data(self):
        """
        Show modal data in a table-like structure.
        """
        print('   Nat. f.      Damping')
        print(23*'-')
        for i, f in enumerate(self.nat_freq):
            print(f'{i+1}) {f:6.1f}\t{self.nat_xi[i]:5.4f}')


def _irfft_adjusted_lower_limit(x, low_lim, indices):
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

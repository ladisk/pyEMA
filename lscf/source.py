import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg
from tqdm import tqdm_notebook as tqdm

class lscf():
    """
    Least-Squares Complex Frequency-domain estimate.
    """
    def __init__(self, frf, freq, lower, upper,pol_order_low,pol_order_high):
        """
        frf - FRF pri vseh lokacijah in frekvencah
        freq - 1D array frekvenc
        lower - najbolj spodnja opazovana frekvenca
        upper - najbolj zgornja opazovana frekvenca
        """
        self.lower = lower
        self.upper = upper
        self.pol_order_low = pol_order_low
        self.pol_order_high = pol_order_high

        self.frf = np.asarray(frf) #frekenčne prenosne funkcije
        self.freq = freq #frekvence (vse!)
        self.omega = 2* np.pi * self.freq #krožne frekvence
        
        self.select_freq = (self.freq >= self.lower) & (self.freq <= self.upper) #omejimo frekvence na zgornjo in spodnjo
        self.sel_clip_up = (self.freq >= 0) & (self.freq <= self.upper)#odstranimo zgornje frekvence
        self.sel_lower = (self.freq >= 0) & (self.freq <= self.lower)#izberemo samo spodnje frekvence
        
        self.freq_sel = self.freq[self.sel_clip_up]
        self.omega_sel = self.omega[self.select_freq]
        
        self.samp_time = 1/(2 * self.freq_sel[-1]-self.freq_sel[0])
        
        
        self.frf_sel = self.frf[::1, self.sel_clip_up]
        self.frf_sel[:, self.sel_lower[:self.frf_sel.shape[1]]] = 0

        self.frf_in = self.frf[:, self.select_freq]
        
    def get_poles(self):
        self.poles = []
        self.f_poles = []
        self.ceta = []
        self.cetaT = []
        for n in tqdm(range(self.pol_order_low, self.pol_order_high)):
            M = np.zeros((n+1, n+1), dtype=complex)
            for Ho_ in self.frf_sel:
                S_fft = -1/(len(Ho_)) * np.real(np.fft.fft(Ho_, 2*len(Ho_)))
                T_fft = 1/(len(Ho_)) * np.real(np.fft.fft(np.abs(Ho_)**2, 2*len(Ho_)))

                S_col = S_fft[:n+1]
                S_row = np.append(S_fft[-n:], S_fft[0])[::-1] 
                T_col = T_fft[:n+1]

                S = scipy.linalg.toeplitz(S_col, S_row) #S ni simetričen
                T = scipy.linalg.toeplitz(T_col, T_col) #T je simetričen
                R = np.identity(np.size(S, axis=0))     #če ne uporabljamo uteži je R enotska matrika

                M += T - S.T @ np.linalg.inv(R) @ S
            
            M = 2 * M

            A = M[:n, :n]
            B = -M[:n, n]

            x = np.linalg.inv(A) @ np.vstack(B)
            est_alfa = np.append(x, 1)
            est_beta = -np.linalg.inv(R) @ S @ est_alfa

            roots = np.roots(est_alfa[::-1]) 
            poles = 1/self.samp_time * (np.log(np.abs(roots)) - 1j * np.angle(roots)) 
            f_pole = np.imag(poles)/(2*np.pi)
            ceta = -np.real(poles) / np.abs(poles)
            cetaT = -np.real(poles) / np.abs(poles)

            
            self.poles.append(poles)
            self.f_poles.append(f_pole)
            self.ceta.append(ceta)
            self.cetaT.append(cetaT)

    def stab_chart(self,poles,fn_temp = 0.001, xi_temp= 0.05, legend = False,latex_render=False, title=None):
        """
        Prikaz stabilizacijskega diagrame.

        Podaja možnost interaktivnega izbiranja polov. Ko izberemo pol se sproti izvaja identifikacija.
        Identifikacijo lahko izvajamo tudi na podlagi že poznanih približkov polov in jih vnesemo direktno v funkcijo `identifikacija()`

        Možnost 1:
        >>> a.stab_chart()
        >>> a.nat_freq #lastne frekvence
        >>> a.nat_ceta #modalno dušenje

        Možnost 2:
        >>> a.stab_chart()
        >>> priblizki = [234, 545]
        >>> a.identification(priblizki)
        >>> a.nat_freq #lastne frekvence
        >>> a.nat_ceta #modalno dušenje
        """
        Nmax = self.pol_order_high-self.pol_order_low
        fn_temp, xi_temp, test_fn, test_xi = stabilisation(poles ,Nmax,err_fn = fn_temp,err_xi =xi_temp)

        fig, ax1 = plt.subplots(figsize = (10,4))
        ax2 = ax1.twinx()
        plt.xlim(self.lower,self.upper)
        
        ax1.set_xlabel(r'$f$ [Hz]', fontsize=12)
        ax1.set_ylabel(r'Polynom order', fontsize=12)
        ax2.set_ylabel(r'$|\alpha|$', fontsize=12)
        
        if latex_render is True:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            ax1.tick_params(axis='both', which='major', labelsize=12)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            ax1.set_xlabel(r'$f$ [Hz]', fontsize=12)
            ax1.set_ylabel(r'Red polinoma', fontsize=12)
            ax2.set_ylabel(r'$|\alpha|_{log}$', fontsize=12)
            ax1.set_xlim([self.lower, self.upper])
        
        a=np.argwhere((test_fn>0) & (test_xi==0)) # stable eigenfrequencues, unstable damping ratios
        b=np.argwhere((test_fn>0) & (test_xi>0) ) # stable eigenfrequencies, stable damping ratios
        c=np.argwhere((test_fn==0) & (test_xi==0)) # unstable eigenfrequencues, unstable damping ratios
        d=np.argwhere((test_fn==0) & (test_xi>0)) # unstable eigenfrequencues, stable damping ratios

        for i in range(0,len(a)):
            p1 = ax1.plot(fn_temp[a[i,0], a[i,1]], 1+a[i,1], c='b', marker='x',markersize=3)
        for j in range(0,len(b)):
            p2 = ax1.plot(fn_temp[b[j,0], b[j,1]] ,1+b[j,1], c='g', marker='x',markersize=4)
        for k in range(0,len(c)):
            p3 = ax1.plot(fn_temp[c[k,0], c[k,1]], 1+c[k,1], c='r', marker='.',markersize=3)
        for l in range(0,len(d)):
            p4 = ax1.plot(fn_temp[d[l,0], d[l,1]], 1+d[l,1], c='r', marker='*',markersize=3)
        
        if legend:
            p1 = ax1.plot(fn_temp[a[i,0], a[i,1]], 1+a[i,1],'x', c='b',markersize=3,label = "stable frequency, unstable damping")
            p2 = ax1.plot(fn_temp[b[j,0], b[j,1]] ,1+b[j,1],'x', c='g',markersize=4,label = "stable frequency, stable damping")
            p3 = ax1.plot(fn_temp[c[k,0], c[k,1]], 1+c[k,1],'.', c='r',markersize=3,label = "unstable frequency, unstable damping")
            p4 = ax1.plot(fn_temp[d[l,0], d[l,1]], 1+d[l,1],'*', c='r',markersize=3,label = "unstable frequency, stable damping")
            ax1.legend(loc = 'best')

        ax2.semilogy(self.freq_sel, np.average(np.abs(self.frf_sel), axis=0), alpha=0.7,color = 'k');
        ax1.grid(True);

        print('Za izbiranje lastnih frekvenc uporabi SREDNJI gumb.\nZa izbris zadnje točke uporabi DESNI gumb.')
        self.nat_freq = []
        self.nat_ceta = []
        self.pole_ind = []
        
        line, = ax1.plot(self.nat_freq, np.repeat(self.pol_order_high, len(self.nat_freq)),'kv',markersize = 8);
        def onclick(event):
            if event.button == 2: #če smo pritisnili gumb 2 (srednji na miški)
                self.identification([event.xdata], self.nat_freq, self.nat_ceta, self.pole_ind) #identifikacija lastnih frekvenc in dušenja
                print(f'{len(self.nat_freq)}. frekvenca: ~{int(np.round(event.xdata))} --> {self.nat_freq[-1]} Hz')
            elif event.button == 3:
                try:
                    del self.nat_freq[-1] #izbrišemo zadnjo točko
                    del self.nat_ceta[-1]
                    del self.pole_ind[-1]
                    print('Izbrisana zadnja točka...')
                except:
                    pass

            line.set_xdata(np.asarray(self.nat_freq)) #posodobimo podatke
            line.set_ydata(np.repeat(Nmax*1.04, len(self.nat_freq)))
            fig.canvas.draw() 

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        if title is not None:
            plt.savefig(title)

    def identification(self, approx_nat_freq, nat_freq=None, nat_ceta=None, pole_ind=None):
        pole_ind = []
        for i, fr in enumerate(approx_nat_freq):            
            sel = np.argmin(np.abs(self.f_poles[-1] - fr))
            pole_ind.append(np.argmin(np.abs(self.f_poles[-1] - self.f_poles[-1][sel])))

        if nat_freq is None and nat_ceta is None:
            self.nat_freq = self.f_poles[-1][pole_ind]
            self.nat_ceta = self.ceta[-1][pole_ind]
            self.pole_ind = pole_ind
        else:
            nat_freq.append(self.f_poles[-1][pole_ind][0])
            nat_ceta.append(self.ceta[-1][pole_ind][0])
            self.pole_ind.append(pole_ind[0])
            self.nat_freq = nat_freq
            self.nat_ceta = nat_ceta

    ######################################################################################
    def lsfd(self, whose_poles='own'):
        ndim = self.frf.ndim
        if whose_poles == 'own':
            poles = self.poles[-1][self.pole_ind]
            n_poles = len(self.pole_ind)
        else:
            poles = whose_poles.poles[-1][whose_poles.pole_ind]
            n_poles = len(whose_poles.pole_ind)
        
        w = np.append(-self.omega_sel[1:][::-1], self.omega_sel[1:])
        alpha = np.append(self.frf_in[:, 1:].conjugate()[:, ::-1], self.frf_in[:, 1:], ndim-1)
        TA = np.ones([len(w), n_poles], complex)
        for n in range(n_poles):
            TA[:, n] = 1/(1j*w - poles[n])
        AT = np.linalg.pinv(TA)

        if ndim == 1:
            A_LSFD = np.dot(AT, self.frf_in)
        elif ndim == 2:
            IO = self.frf_in.shape[0]
            A_LSFD = np.zeros([IO, n_poles], complex)
            for v in range(IO):
                A_LSFD[v, :] = np.dot(AT, alpha[v, :])
        self.A_LSFD = A_LSFD
        self.poles = poles
        return A_LSFD

    def FRF_reconstruct(self, FRF_ind):
        FRF_true = np.zeros(len(self.omega), complex)
        for n in range(self.A_LSFD.shape[1]-2):
            FRF_true += (self.A_LSFD[FRF_ind, n]/(1j*self.omega - self.poles[n]))
        FRF_true += self.A_LSFD[FRF_ind, -2]/(self.omega**2) + self.A_LSFD[FRF_ind, -1]
        return FRF_true
    ######################################################################################
    
    def modal_const(self, frf_loc, whos_poles='own', whos_inds='own', form = 'accelerance'):
        self.frf_loc = frf_loc
        if whos_poles =='own':
            poles = self.poles[-1][self.pole_ind]
            n_poles = len(self.pole_ind)
        else:
            poles = whos_poles.poles[-1][whos_poles.pole_ind]
            n_poles = len(whos_poles.pole_ind)

        poles_mk = np.concatenate((poles, np.conj(poles)))

        mk1 = self.frf[self.frf_loc, self.select_freq]
        mk2 = np.zeros((len(self.omega_sel), 2*n_poles+2), dtype=complex)
        for i in range(2*n_poles+2):
            if i<2*n_poles:
                mk2[:, i] = 1/(1j * self.omega_sel - poles_mk[i])
            elif i==2*n_poles:
                mk2[:, i] = -1/(self.omega_sel**2) #lower residual
            elif i==2*n_poles+1:
                mk2[:, i] = 1 #upper residual


        mk2_m = np.ma.masked_invalid(mk2)
        mk2_mask = np.ma.getmask(mk2_m)
        mk2_m[mk2_mask] = 0

        self.mk2 = np.copy(mk2_m)
        self.mk3 = np.linalg.pinv(self.mk2) @ mk1.T
    
    def reconstruct(self, whos_poles='own', whos_inds='own'):
        if whos_poles =='own':
            poles = self.poles[-1][self.pole_ind]
            n_poles = len(self.pole_ind)
        else:
            poles = whos_poles.poles[-1][whos_poles.pole_ind]
            n_poles = len(whos_poles.pole_ind)
            
        poles_mk = np.concatenate((poles, np.conj(poles)))
        
        rek2 = np.zeros((len(self.omega[self.sel_clip_up]), 2*n_poles+2), dtype=complex)
        for i in range(2*n_poles+2):
            if i<2*n_poles:
                rek2[:, i] = 1/(1j * self.omega[self.sel_clip_up] - poles_mk[i])
            elif i==2*n_poles:
                rek2[:, i] = -1/(self.omega[self.sel_clip_up]**2)
            elif i==2*n_poles+1:
                rek2[:, i] = 1
                
        self.rek = rek2 @ self.mk3 #rekonstrukcija
        self.rekf = self.freq[self.sel_clip_up] #freq vektor rekonstrukcije

        
def complex_freq_to_freq_and_damp(sr):
    """
    Convert the complex natural frequencies to natural frequencies and the
    corresponding dampings.

    :param sr: complex natural frequencies
    :return: natural frequency and damping
    """

    fr = np.abs(sr)
    xir = -sr.real/fr
    fr /= (2 * np.pi)

    return fr, xir

def redundant_values(omega, xi, prec):
    """
    This function supresses the redundant values of frequency and damping
    vectors, which are the consequence of conjugate values

    Input:
        omega - eiqenfrquencies vector
        xi - damping ratios vector
        prec - absoulute precision in order to distinguish between two values

    @author: Blaz Starc
    @contact: blaz.starc@fs.uni-lj.si
    """
    N = len(omega)
    test_omega = np.zeros((N,N), dtype='int')
    for i in range(1,N):
        for j in range(0,i):
            if np.abs((omega[i] - omega[j])) < prec:
                test_omega[i,j] = 1
            else: test_omega[i,j] = 0
    test = np.zeros(N, dtype = 'int')
    for i in range(0,N):
        test[i] = np.sum(test_omega[i,:])
    
    omega_mod = omega[np.argwhere(test<1)]
    xi_mod = xi[np.argwhere(test<1)]
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
    
    @author: Blaz Starc
    @contact: blaz.starc@fs.uni-lj.si
    """

    # TODO: check this later for optimisation # this doffers by LSCE and LSCF
    fn_temp = np.zeros((2*nmax, nmax), dtype = 'double')
    xi_temp = np.zeros((2*nmax, nmax), dtype = 'double')
    test_fn = np.zeros((2*nmax, nmax), dtype = 'int')
    test_xi = np.zeros((2*nmax, nmax), dtype = 'int')

    for nr, n in enumerate(range(nmax)):
        fn, xi = complex_freq_to_freq_and_damp(sr[nr])
        fn, xi = redundant_values(fn, xi, 1e-3) # elimination of conjugate values in
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
            fn_test = np.zeros((len(fn), len(fn_temp[:, n - 1])), dtype ='int')
            for i in range(0, len(fn)):
                for j in range(0, len(fn_temp[0:2*(n), n-1])):
                    if fn_temp[j, n-2] ==  0:
                        fn_test[i,j] = 0
                    else:
                        if np.abs((fn[i] - fn_temp[j, n-2])/fn_temp[j, n-2]) < err_fn:
                            fn_test[i,j] = 1
                        else: fn_test[i,j] = 0

            for i in range(0, len(fn)):
                test_fn[i, n - 1] = np.sum(fn_test[i, :]) # all rows are summed together

            # The same procedure as for eigenfrequencies is applied for damping
            xi_test = np.zeros((len(xi), len(xi_temp[:, n - 1])), dtype ='int')
            for i in range(0, len(xi)):
                for j in range(0, len(xi_temp[0:2*(n), n-1])):
                    if xi_temp[j, n-2]==0:
                        xi_test[i,j] = 0
                    else:
                        if np.abs((xi[i] - xi_temp[j, n-2])/xi_temp[j, n-2]) < err_xi:
                            xi_test[i,j] = 1
                        else: xi_test[i,j] = 0
            for i in range(0, len(xi)):
                test_xi[i, n - 1] = np.sum(xi_test[i, :])

            # If the frequency/damping values corresponded to the previous iteration,
            # a mean of the two values is computed, otherwise the value stays the same
            for i in range(0, len(fn)):
                for j in range(0, len(fn_temp[0:2*(n), n-1])):
                    if fn_test[i,j] == 1:
                        fn_temp[i, n - 1] = (fn[i] + fn_temp[j, n - 2]) / 2
                    elif fn_test[i,j] == 0:
                        fn_temp[i, n - 1] = fn[i]
            for i in range(0, len(fn)):
                for j in range(0, len(fn_temp[0:2*(n), n-1])):
                    if xi_test[i,j] == 1:
                        xi_temp[i, n - 1] = (xi[i] + xi_temp[j, n - 2]) / 2
                    elif xi_test[i,j] == 0:
                        xi_temp[i, n - 1] = xi[i]

    return fn_temp, xi_temp, test_fn, test_xi
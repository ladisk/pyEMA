import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import stabilization

class SelectPoles:
    def __init__(self, Model):
        """
        Plot the measured Frequency Response Functions and computed poles.

        Picking the poles is done by pressing the SHIFT key + left mouse button.
        To unselect last pole: press SHIFT + right mouse button.
        To unselect closest pole: press SHIFT + middle mouse button.

        For more information check the HELP menu tab in the chart window.

        param model: object of pyEMA.Model
        """
        self.Model = Model
        self.shift_is_held = False
        self.chart_type = 0 # 0 - stability chart, 1 - cluster diagram
        self.show_legend = 0
        self.frf_plot_type = 'abs'

        self.Model.nat_freq = []
        self.Model.nat_xi = []
        self.Model.pole_ind = []
        
    
        self.root = tk.Tk()
        self.root.title('Stability Chart')
        self.fig = Figure(figsize=(20, 8))

        # Create axes
        self.ax2 = self.fig.add_subplot(111)
        self.ax1 = self.ax2.twinx()
        self.ax1.grid(True)
        
        # Tkinter menu
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Save figure', command=self.save_this_figure)
        menubar.add_cascade(label='File', menu=filemenu)

        chartmenu = tk.Menu(menubar, tearoff=0)
        chartmenu.add_command(label='Stability chart', command=lambda: self.toggle_chart_type(0))
        chartmenu.add_command(label='Cluster diagram', command=lambda: self.toggle_chart_type(1))
        menubar.add_cascade(label="Chart type", menu=chartmenu)

        mifmenu = tk.Menu(menubar, tearoff=0)
        mifmenu.add_command(label='Plot mean abs', command=lambda: self.toggle_mif_frf('abs'))
        mifmenu.add_command(label='Plot all FRFs', command=lambda: self.toggle_mif_frf('all'))
        menubar.add_cascade(label="FRF plot type", menu=mifmenu)

        legendmenu = tk.Menu(menubar, tearoff=0)
        legendmenu.add_command(label='Show legend', command=lambda: self.toggle_legend(1))
        legendmenu.add_command(label='Hide legend', command=lambda: self.toggle_legend(0))
        menubar.add_cascade(label="Legend", menu=legendmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label='Help', command=self.show_help)
        menubar.add_cascade(label='Help', menu=helpmenu)

        self.root.config(menu=menubar)


        # Program execution
        self.plot_frf(initial=True)
        self.get_stability()
        self.plot_stability()

        # Integrate matplotlib figure
        canvas = FigureCanvasTkAgg(self.fig, self.root)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        NavigationToolbar2Tk(canvas, self.root)

        
        # Connecting functions to event manager
        self.fig.canvas.mpl_connect('key_press_event', lambda x: self.on_key_press(x))
        self.fig.canvas.mpl_connect('key_release_event', lambda x: self.on_key_release(x))
        self.fig.canvas.mpl_connect('button_press_event', lambda x: self.on_click(x))

        self.root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing())
        self.root.mainloop()


    def plot_frf(self, initial=False):
        """Reconstruct and plot the Frequency Response Function.

        This is done on the fly.

        :param initial: if True, the FRF is not computed, only the measured
            FRFs are shown.
        """
        self.ax2.clear()
        if self.frf_plot_type == 'abs':
            self.ax2.semilogy(self.Model.freq, np.average(
                    np.abs(self.Model.frf), axis=0), alpha=0.7, color='k')
        elif self.frf_plot_type == 'all':
            self.ax2.semilogy(self.Model.freq, np.abs(self.Model.frf.T), alpha=0.3, color='k')

        if not initial and len(self.Model.nat_freq) > 0:
            self.H, self.A = self.Model.get_constants(whose_poles='own', FRF_ind='all', least_squares_type='new')
            if self.frf_plot_type == 'abs':
                self.ax2.semilogy(self.Model.freq, np.average(
                    np.abs(self.H), axis=0), color='r', lw=2)
            elif self.frf_plot_type == 'all':
                self.ax2.semilogy(self.Model.freq, np.abs(self.H.T), color='r', lw=1)
        
        else:
            x_position = (self.Model.lower + self.Model.upper) / 2
            y_position = np.max(np.abs(self.Model.frf))
            message = [
                'Select a pole: SHIFT + LEFT mouse button',
                'Unselect a pole: SHIFT + RIGHT mouse button'
            ]
            self.ax2.text(x_position, y_position, '\n'.join(message), 
                fontsize=12, verticalalignment='top', horizontalalignment='center',
                bbox=dict(facecolor='lightgreen', edgecolor='lightgreen'))
        
        self.ax1.set_xlim([self.Model.lower, self.Model.upper])
        self.fig.canvas.draw()
    

    def get_stability(self, fn_temp=0.001, xi_temp=0.05):
        """Get the stability matrix.
        
        :param fn_temp: Natural frequency stability criterion.
        :param xi_temp: Damping stability criterion.
        """
        Nmax = self.Model.pol_order_high
        self.fn_temp, self.xi_temp, self.test_fn, self.test_xi = stabilization._stabilization(
            self.Model.all_poles, Nmax, err_fn=fn_temp, err_xi=xi_temp)


    def plot_stability(self, update_ticks=False):
        if not update_ticks:
            self.ax1.clear()
            self.ax1.grid(True)

            # stable eigenfrequencies, unstable damping ratios
            a = np.argwhere((self.test_fn > 0) & ((self.test_xi == 0) | (self.xi_temp <= 0)))
            # stable eigenfrequencies, stable damping ratios
            b = np.argwhere((self.test_fn > 0) & ((self.test_xi > 0) & (self.xi_temp > 0)))
            # unstable eigenfrequencies, unstable damping ratios
            c = np.argwhere((self.test_fn == 0) & ((self.test_xi == 0) | (self.xi_temp <= 0)))
            # unstable eigenfrequencies, stable damping ratios
            d = np.argwhere((self.test_fn == 0) & ((self.test_xi > 0) & (self.xi_temp > 0)))

            p1 = self.ax1.plot(self.fn_temp[a[:, 0], a[:, 1]], 1+a[:, 1], 'bx',
                        markersize=4, label="stable frequency, unstable damping")
            p2 = self.ax1.plot(self.fn_temp[b[:, 0], b[:, 1]], 1+b[:, 1], 'gx',
                        markersize=7, label="stable frequency, stable damping")
            p3 = self.ax1.plot(self.fn_temp[c[:, 0], c[:, 1]], 1+c[:, 1], 'r.',
                        markersize=4, label="unstable frequency, unstable damping")
            p4 = self.ax1.plot(self.fn_temp[d[:, 0], d[:, 1]], 1+d[:, 1], 'r*',
                        markersize=4, label="unstable frequency, stable damping")
            
            self.line, = self.ax1.plot(self.Model.nat_freq, np.repeat(
                    self.Model.pol_order_high, len(self.Model.nat_freq)), 'kv', markersize=8)
            self.selected, = self.ax1.plot([self.Model.pole_freq[p[0]][p[1]] for p in self.Model.pole_ind], 
                                            [p[0] for p in self.Model.pole_ind], 'ko')
            
            if self.show_legend:
                self.pole_legend = self.ax1.legend(loc='upper center', ncol=2, frameon=True)

            self.ax1.set_title('Stability chart')

            
            self.ax1.set_ylabel('Polynomial order')

            plt.tight_layout()
        else:
            self.line.set_xdata(np.asarray(self.Model.nat_freq))  # update data
            self.line.set_ydata(np.repeat(self.Model.pol_order_high*1.04, len(self.Model.nat_freq)))

            self.selected.set_xdata([self.Model.pole_freq[p[0]][p[1]]
                                for p in self.Model.pole_ind])  # update data
            self.selected.set_ydata([p[0] for p in self.Model.pole_ind])

            plt.tight_layout()

        self.ax1.set_ylim([0, self.Model.pol_order_high+5])
        self.fig.canvas.draw()


    def plot_cluster(self, update_ticks=False):
        """Plot clusters - damping with respect to frequency.

        :param update_ticks: if True, only ticks are updated, not the whole plot.
        """
        b1 = np.argwhere(((self.test_fn > 0) & ((self.test_xi > 0) & (self.xi_temp > 0))) & ((self.fn_temp > self.Model.lower) & (self.fn_temp < self.Model.upper)))
        
        if not update_ticks:
            self.ax1.clear()
            self.ax1.grid(True)

            # stable eigenfrequencies, unstable damping ratios
            a = np.argwhere((self.test_fn > 0) & ((self.test_xi == 0) | (self.xi_temp <= 0)))
            # stable eigenfrequencies, stable damping ratios
            b = np.argwhere((self.test_fn > 0) & ((self.test_xi > 0) & (self.xi_temp > 0)))
            # unstable eigenfrequencies, unstable damping ratios
            c = np.argwhere((self.test_fn == 0) & ((self.test_xi == 0) | (self.xi_temp <= 0)))
            # unstable eigenfrequencies, stable damping ratios
            d = np.argwhere((self.test_fn == 0) & ((self.test_xi > 0) & (self.xi_temp > 0)))

            p1 = self.ax1.plot(self.fn_temp[a[:, 0], a[:, 1]], self.xi_temp[a[:, 0], a[:, 1]], 'bx',
                        markersize=4, label="stable frequency, unstable damping")
            p2 = self.ax1.plot(self.fn_temp[b[:, 0], b[:, 1]], self.xi_temp[b[:, 0], b[:, 1]], 'gx',
                        markersize=7, label="stable frequency, stable damping")
            p3 = self.ax1.plot(self.fn_temp[c[:, 0], c[:, 1]], self.xi_temp[c[:, 0], c[:, 1]], 'r.',
                        markersize=4, label="unstable frequency, unstable damping")
            p4 = self.ax1.plot(self.fn_temp[d[:, 0], d[:, 1]], self.xi_temp[d[:, 0], d[:, 1]], 'r*',
                        markersize=4, label="unstable frequency, stable damping")
            
            self.line, = self.ax1.plot(self.Model.nat_freq, np.repeat(
                    1.05*np.max(self.xi_temp[b1[:, 0], b1[:, 1]]), len(self.Model.nat_freq)), 'kv', markersize=8)
            self.selected, = self.ax1.plot([self.Model.pole_freq[p[0]][p[1]] for p in self.Model.pole_ind], 
                                            [self.Model.pole_xi[p[0]][p[1]] for p in self.Model.pole_ind], 'ko')
            
            if self.show_legend:
                self.pole_legend = self.ax1.legend(loc='upper center', ncol=2, frameon=True)

            self.ax1.set_title('Cluster diagram')

            self.ax1.set_ylabel('Damping ratio')
            plt.tight_layout()
        else:
            self.line.set_xdata(np.asarray(self.Model.nat_freq))  # update data
            self.line.set_ydata(np.repeat(1.05*np.max(self.xi_temp[b1[:, 0], b1[:, 1]]), len(self.Model.nat_freq)))

            self.selected.set_xdata([self.Model.pole_freq[p[0]][p[1]] for p in self.Model.pole_ind])  # update data
            self.selected.set_ydata([self.Model.pole_xi[p[0]][p[1]] for p in self.Model.pole_ind])
        
        to_lim = self.xi_temp[b1[:, 0], b1[:, 1]]
        up_lim = min(np.max(to_lim), np.mean(to_lim)+2*np.std(to_lim))
        self.ax1.set_ylim([np.mean(to_lim)-np.std(to_lim), up_lim])
        self.fig.canvas.draw()


    def get_closest_poles_stability(self):
        """
        On-the-fly selection of the closest poles.        
        """
        y_ind = int(np.argmin(np.abs(np.arange(0, len(self.Model.pole_freq)
                                               )-self.y_data_pole)))  # Find closest pole order
        # Find cloeset frequency
        sel = np.argmin(np.abs(self.Model.pole_freq[y_ind] - self.x_data_pole))

        self.Model.pole_ind.append([y_ind, sel])
        self.Model.nat_freq.append(self.Model.pole_freq[y_ind][sel])
        self.Model.nat_xi.append(self.Model.pole_xi[y_ind][sel])
        self.sort_selected_poles()


    def get_closest_poles_cluster(self):
        """
        On-the-fly selection of the closest poles.        
        """
        min_ = []
        for y_ind in range(len(self.Model.pole_freq)):
            for sel in range(len(self.Model.pole_freq[y_ind])):
                min_.append([y_ind, sel, np.abs(self.Model.pole_freq[y_ind][sel]-self.x_data_pole), np.abs(self.Model.pole_xi[y_ind][sel]-self.y_data_pole[0])])

        min_a = np.asarray(min_)
        # select the poles that have a frequency within 2% of the observed range
        mask = min_a[:, 2] < (self.Model.upper - self.Model.lower) * 0.02
        min_ind_x = np.argmin(min_a[mask][:, 3])
        
        y_ind = int(min_a[mask][min_ind_x, 0])
        sel = int(min_a[mask][min_ind_x, 1])

        self.Model.pole_ind.append([y_ind, sel])
        self.Model.nat_freq.append(self.Model.pole_freq[y_ind][sel])
        self.Model.nat_xi.append(self.Model.pole_xi[y_ind][sel])
        self.sort_selected_poles()
    

    def on_click(self, event):
        # on button 1 press (left mouse button) + SHIFT is held
        if event.button == 1 and self.shift_is_held:
            self.y_data_pole = [event.ydata]
            self.x_data_pole = event.xdata
            if self.chart_type == 0:
                self.get_closest_poles_stability()
            elif self.chart_type == 1:
                self.get_closest_poles_cluster()

            self.plot_frf()
        
        # On button 3 press (left mouse button)
        elif event.button == 3 and self.shift_is_held:
            try:
                del self.Model.nat_freq[-1]  # delete last point
                del self.Model.nat_xi[-1]
                del self.Model.pole_ind[-1]
                self.plot_frf()
            except:
                pass

        elif event.button == 2 and self.shift_is_held:
            i = np.argmin(np.abs(self.Model.nat_freq - event.xdata))
            try:
                del self.Model.nat_freq[i]
                del self.Model.nat_xi[i]
                del self.Model.pole_ind[i]
                self.plot_frf()
            except:
                pass


        if self.shift_is_held:
            if self.chart_type == 0:
                self.plot_stability(update_ticks=True)
            elif self.chart_type == 1:
                self.plot_cluster(update_ticks=True)


    def on_key_press(self, event):
        """Function triggered on key press (SHIFT)."""
        if event.key == 'shift':
            self.shift_is_held = True
    

    def on_key_release(self, event):
        """Function triggered on key release (SHIFT)."""
        if event.key == 'shift':
            self.shift_is_held = False
    

    def on_closing(self):
        self.root.destroy()

    
    def toggle_legend(self, x):
        if x:
            self.show_legend = 1
        else:
            self.show_legend = 0

        if self.chart_type == 0:
            self.plot_stability()
        elif self.chart_type == 1:
            self.plot_cluster()
    
    
    def toggle_chart_type(self, x):
        if x == 0:
            self.chart_type = 0
            self.plot_stability()
        elif x == 1:
            self.chart_type = 1
            self.plot_cluster()
        
    
    def toggle_mif_frf(self, x):
        self.frf_plot_type = x
        self.plot_frf()


    def sort_selected_poles(self):
        _ = np.argsort(self.Model.nat_freq)
        self.Model.pole_ind = list(np.array(self.Model.pole_ind)[_])
        self.Model.nat_freq = list(np.array(self.Model.nat_freq)[_])
        self.Model.nat_xi = list(np.array(self.Model.nat_xi)[_])


    def show_help(self):
        lines = [
            'Pole selection help',
            ' ',
            '- Select a pole: SHIFT + LEFT mouse button',
            '- Unselect a pole: SHIFT + RIGHT mouse button',
            '- Unselect the closest pole (frequency wise): SHIFT + MIDDLE mouse button',
            ' ',
            '- Two different types of charts are currently avaliable:',
            '  1. stability chart, where pole frequencies are plotted against polynomial orders',
            '  2. cluster diagram, where pole frequencies are plotted against damping ratios'
        ]
        tk.messagebox.showinfo('Picking poles', '\n'.join(lines))


    def save_this_figure(self):
        filename = 'pole_chart_'
        directory = 'pole_figures'

        if not os.path.exists(directory):
            os.mkdir(directory)

        files = glob.glob(directory + '/*.png')
        i = 1
        while True:
            f = os.path.join(directory, filename + f'{i:0>3}.png')
            if f not in files:
                break
            i += 1

        self.fig.savefig(f)




"""The fit is a goodness measure of the EMC algorithm. This module
plots the history of it."""
import numpy
import module_template
import embedded_matplotlib

class FitData(module_template.Data):
    """Reads the fit and best rotation fit and returns it on request."""
    def __init__(self):
        super(FitData, self).__init__()
        self._fit_data = None
        self._fit_data_best_rot = None

        self.read_fit()

    def get_fit(self, reload_data=False):
        """Returns the fit. Reload if requested."""
        if reload_data:
            self.read_fit()
        return self._fit_data

    def get_fit_best_rot(self, reload_data=False):
        """Returns the fit of the best fitting rotations. Reload if requested."""
        if reload_data:
            self.read_fit()
        return self._fit_data_best_rot

    def read_fit(self):
        """Reload the data."""
        sucess = False
        try:
            raw_data = numpy.loadtxt('output/fit.data')
            if len(raw_data.shape) == 1:
                raw_data = raw_data.reshape(1, len(raw_data))
            self._fit_data = raw_data.mean(axis=1)
            sucess = True
        except IOError:
            print "ioerror"
            self.read_error.emit()
        try:
            self._fit_data_best_rot = numpy.loadtxt('output/fit_best_rot.data').mean(axis=1)
            sucess = True
        except IOError:
            print "ioerror"
            self.read_error.emit()
        if sucess:
            self.data_changed.emit()

class FitViewer(module_template.Viewer):
    """Plot the fit using matplotlib."""
    def __init__(self):
        super(FitViewer, self).__init__()
        self._fit_plot = None
        self._fit_best_rot_plot = None
        self._iteration_plot = None

        self._widget, self._fig, self._canvas, self._mpl_toolbar = embedded_matplotlib.get_matplotlib_widget()
        self._axes = self._fig.add_subplot(111)

    def plot_fit(self, likelihood, likelihood_best_rot=None, iteration=None):
        """Replace the current plot with the provided data."""
        self._axes.clear()
        self._fit_plot = self._axes.plot(likelihood, color='black', lw=2, label='Average fit')
        if not likelihood_best_rot is None:
            self._fit_best_rot_plot = self._axes.plot(likelihood_best_rot, color='red',
                                                      lw=2, label='Best rotation fit')
        self._axes.set_ylim((0., 1.))
        self._axes.legend()
        if not iteration is None:
            limits = self._axes.get_ylim()
            self._iteration_plot = self._axes.plot([iteration]*2, limits, color='green', lw=2)[0]
        self._canvas.draw()

    def set_iteration(self, iteration):
        """Move the iteration indicator."""
        self._iteration_plot.set_xdata([iteration]*2)
        self._canvas.draw()

class FitControll(module_template.Controll):
    """This class contains a widget for the user input. It also contains the Data and View
    and the communication between them."""
    def __init__(self, common_controll, viewer, data):
        super(FitControll, self).__init__(common_controll, viewer, data)

    def draw_hard(self):
        """Draw the widget. In practice call draw() instead as it only draws when the
        module is active."""
        self._viewer.plot_fit(self._data.get_fit(reload_data=True), self._data.get_fit_best_rot(reload_data=False),
                              self._common_controll.get_iteration())


class Plugin(module_template.Plugin):
    """Collects all classes of the module."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = FitViewer()
        self._data = FitData()
        self._controll = FitControll(common_controll, self._viewer, self._data)


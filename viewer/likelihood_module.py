"""Plots the likelihood history."""
import numpy
import module_template
import embedded_matplotlib

class LikelihoodData(module_template.Data):
    """Reads the likelihood and returns it on request."""
    def __init__(self):
        super(LikelihoodData, self).__init__()
        self._likelihood_data = None

        self.read_likelihood()

    def get_likelihood(self, reload_data=False):
        """Return the likelihood. Reload if requested. Otherwise a reload can be
        forced by the read_likelihood function."""
        if reload_data:
            self.read_likelihood()
        return self._likelihood_data

    def read_likelihood(self):
        """Reload the likelihood to make it up to date with lateset update."""
        try:
            self._likelihood_data = numpy.loadtxt('output/likelihood.data')
        except IOError:
            try:
                self._likelihood_data = numpy.loadtxt('likelihood.data')
            except IOError:
                self.read_error.emit()
            self.read_error.emit()
        self.data_changed.emit()

class LikelihoodViewer(module_template.Viewer):
    """Uses matplotlib to display the likelihood history."""
    def __init__(self):
        super(LikelihoodViewer, self).__init__()
        self._likelihood_plot = None
        self._iteration_plot = None

        self._widget, self._fig, self._canvas, self._mpl_toolbar = embedded_matplotlib.get_matplotlib_widget()
        self._axes = self._fig.add_subplot(111)

    def plot_likelihood(self, likelihood, iteration=None):
        """Replace the current plot with the the provided data"""
        self._axes.clear()
        self._likelihood_plot = self._axes.plot(likelihood, color='black', lw=2)
        if iteration != None:
            limits = self._axes.get_ylim()
            self._iteration_plot = self._axes.plot([iteration]*2, limits, color='green', lw=2)[0]
        self._canvas.draw()

    def set_iteration(self, iteration):
        """Move the iteration indicator."""
        self._iteration_plot.set_xdata([iteration]*2)
        self._canvas.draw()

class LikelihoodControll(module_template.Controll):
    """This class contains a widget for the user input. It also contains the Data and View
    and the communication between them."""
    def __init__(self, common_controll, viewer, data):
        super(LikelihoodControll, self).__init__(common_controll, viewer, data)

    def draw_hard(self):
        """Draw the widget. In practice call draw() instead as it only draws when the
        module is active."""
        self._viewer.plot_likelihood(self._data.get_likelihood(reload_data=True),
                                     self._common_controll.get_iteration())
        #self._viewer.plot_likelihood(self._data.get_likelihood(), self._common_controll.get_iteration())
        #self._viewer.set_iteration(self._common_controll.get_iteration())


class Plugin(module_template.Plugin):
    """Collects all classes of the module."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = LikelihoodViewer()
        self._data = LikelihoodData()
        self._controll = LikelihoodControll(common_controll, self._viewer, self._data)

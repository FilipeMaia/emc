"""Plot the best_scaling output by EMC. This shows the scaling for each
diffraction pattern at the rotation with the highest responsability."""

import module_template
import embedded_matplotlib
import h5py
import numpy

class ScalingData(module_template.Data):
    """Reads the data and returns it on request."""
    def __init__(self):
        super(ScalingData, self).__init__()

    def get_scaling(self, max_iterations=None):
        """Reads and returns the scaling data."""
        if max_iterations < 2:
            max_iterations = None
        try:
            with h5py.File("output/best_scaling.h5", "r") as scaling_handle:
                return scaling_handle["scaling"][:max_iterations, :]
        except IOError:
            #self.read_error.emit()
            return numpy.zeros((2, max_iterations))

class ScalingViewer(module_template.Viewer):
    """Plots all the scalings in one matplotlib view."""
    def __init__(self):
        super(ScalingViewer, self).__init__()
        self._scaling_plot = None
        self._iteration_plot = None

        self._widget, self._fig, self._canvas, self._mpl_toolbar = embedded_matplotlib.get_matplotlib_widget()
        self._axes = self._fig.add_subplot(111)

    def plot_scaling(self, scaling, iteration=None):
        """Change the data being plotted to the one provided. Each row of the input should be one set of
        scaling values and the columns indicate time history."""
        self._axes.clear()
        self._scaling_plot = self._axes.plot(scaling, color='black', lw=1)
        if iteration != None:
            limits = self._axes.get_ylim()
            self._iteration_plot = self._axes.plot([iteration]*2, limits, color='green', lw=2)[0]
        self._canvas.draw()

    def set_iteration(self, iteration):
        """Change the position of the iteration indicator."""
        try:
            self._iteration_plot.set_xdata([iteration]*2)
        except NameError:
            return
        self._canvas.draw()

class ScalingControll(module_template.Controll):
    """This class contains a widget for the user input. It also contains the Data and View
    and the communication between them."""
    def __init__(self, common_controll, viewer, data):
        super(ScalingControll, self).__init__(common_controll, viewer, data)
        #self._setup_gui()
        # self.load_and_draw()
        # self._data.data_changed.connect(self.load_and_draw)

    # def _setup_gui(self):
    #     reload_button = QtGui.QPushButton("Reload")
    #     reload_button.pressed.connect(self._data.read_likelihood)
    #     layout = QtGui.QVBoxLayout()
    #     layout.addWidget(reload_button)
    #     # layout.addStretch()
    #     self.setLayout(layout)

    # def load_and_draw(self):
    #     self._viewer.plot_likelihood(self._data.get_likelihood(), self._common_controll.get_iteration())

    def draw_hard(self):
        """Draw the widget. In practice call draw() instead as it only draws when the
        module is active."""
        iteration = self._common_controll.get_iteration()
        #self._viewer.plot_scaling(self._data.get_all_scalings(iteration), iteration)
        self._viewer.plot_scaling(self._data.get_scaling(self._common_controll.get_max_iterations()),
                                  iteration)

        #self._viewer.plot_likelihood(self._data.get_likelihood(), self._common_controll.get_iteration())
        #self._viewer.set_iteration(self._common_controll.get_iteration())

class Plugin(module_template.Plugin):
    """Collects all classes of the module."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = ScalingViewer()
        self._data = ScalingData()
        self._controll = ScalingControll(common_controll, self._viewer, self._data)



"""This module plots the diffraction patterns used as input for EMC."""
import numpy
from QtVersions import QtCore, QtGui
import module_template
import embedded_matplotlib
#import spimage
import sphelper
import convenient_widgets
import os
import re
from matplotlib.colors import LogNorm

class ImageData(module_template.Data):
    """Provides the diffraction pattern data."""
    number_of_images_changed = QtCore.Signal(int)
    def __init__(self):
        super(ImageData, self).__init__()
        self._images = {}
        self._number_of_images = 0
        self.determine_number_of_images()

    def reset(self):
        """Tells the object that something changed so we can no longer use cached images."""
        self.determine_number_of_images()
        # for key in self._images.keys():
        #     spimage.sp_image_free(self._images[key])
        self._images = {}

    def determine_number_of_images(self):
        """Count the number of diffraction patterns available. Store it withing the class."""
        output = os.listdir("output")
        image_files = [image_name for image_name in output if re.search("^image_[0-9]{4}.h5$", image_name)]
        new_number = len(image_files)
        if new_number != self._number_of_images:
            self._number_of_images = new_number
            self.number_of_images_changed.emit(new_number)

    def get_image(self, index, reload_data=False):
        """Get image of certain index."""
        if index >= self._number_of_images:
            raise ValueError("Image index out of range. %d is above %d" % (index, self._number_of_images))
        if index in self._images.keys() and not reload_data:
            return abs(self._images[index])
        self._images[index] = sphelper.import_spimage("output/image_%.4d.h5" % index, ["image"])
        return abs(self._images[index])

    def get_number_of_images(self):
        """Get the number of images."""
        return self._number_of_images

class ImageViewer(module_template.Viewer):
    """Provide a single image view."""
    def __init__(self):
        super(ImageViewer, self).__init__()

        self._widget, self._fig, self._canvas, self._mpl_toolbar = embedded_matplotlib.get_matplotlib_widget()
        self._axes = self._fig.add_subplot(111)
        self._imshow_object = None

    def plot_image(self, image, log_scale=False):
        """Update the plot to show the provided image."""
        self._axes.clear()
        if log_scale:
            norm = LogNorm()
        else:
            norm = None
        self._imshow_object = self._axes.imshow(image, norm=norm, interpolation='nearest')
        self._canvas.draw()

class ImageControll(module_template.Controll):
    """This class contains a widget for the user input. It also contains the Data and View
    and the communication between them."""
    def __init__(self, common_controll, viewer, data):
        super(ImageControll, self).__init__(common_controll, viewer, data)
        self._index = 0
        self._log_scale = False
        self._image_index_chooser = None

        self._setup_gui()

    def _setup_gui(self):
        """Create the gui for the widget."""
        layout = QtGui.QVBoxLayout()
        self._image_index_chooser = convenient_widgets.IntegerControll(0, max_lim=self._data.get_number_of_images())
        self._image_index_chooser.valueChanged.connect(self.set_image)
        self._data.number_of_images_changed.connect(self._image_index_chooser.set_max)
        layout.addWidget(self._image_index_chooser)

        log_scale_box = QtGui.QCheckBox()
        log_scale_label = QtGui.QLabel("Log Scale")
        log_scale_layout = QtGui.QHBoxLayout()
        log_scale_layout.addWidget(log_scale_box)
        log_scale_layout.addWidget(log_scale_label)
        log_scale_layout.addStretch()
        log_scale_box.stateChanged.connect(self.set_log_scale)
        layout.addLayout(log_scale_layout)

        self._widget.setLayout(layout)

    def set_log_scale(self, state):
        """Set wether or not to use log scale for plotting."""
        self._log_scale = state
        self.draw()

    def set_image(self, index):
        """Set what image to view."""
        self._index = index
        self.draw()

    def on_dir_change(self):
        """Call when something changed so we can no longer use cached data."""
        #self._data.determine_number_of_images()
        self._data.reset()
        self.draw()

    def draw_hard(self):
        """Draw the widget. In practice call draw() instead as it only draws when the
        module is active."""
        try:
            image = self._data.get_image(self._index)
        except ValueError:
            image = numpy.zeros((10, 10))
        self._viewer.plot_image(image, log_scale=self._log_scale)


class Plugin(module_template.Plugin):
    """Collects all classes of the module."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = ImageViewer()
        self._data = ImageData()
        self._controll = ImageControll(common_controll, self._viewer, self._data)

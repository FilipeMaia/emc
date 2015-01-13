"""The templates in this module should be inherited when building a new module
for the viewer. Refer for example to the modelmap_module for reference on how."""
from QtVersions import QtCore, QtGui

class Data(QtCore.QObject):
    """This class provides the data to the program."""
    data_changed = QtCore.Signal()
    read_error = QtCore.Signal()
    properties_changed = QtCore.Signal()
    def __init__(self):
        super(Data, self).__init__()

    def initialize(self):
        """Any initialization that should be done after the viewer is
        properly set up with a window to draw into."""
        pass

class Viewer(QtCore.QObject):
    """This class takes care of the plotting. It is not a widget but provides one
    through the get_widget() function."""
    def __init__(self, parent=None):
        super(Viewer, self).__init__(parent)
        self._widget = QtGui.QWidget(parent)

    def initialize(self):
        """Any initialization that should be done after the viewer is
        properly set up with a window to draw into."""
        pass

    def get_widget(self):
        """Returns the plotting widget."""
        return self._widget

    def save_image(self, file_name):
        """Save the current view to file. If the module doesn't implement it
        a popup message will tell the user so."""
        # message_box = QtGui.QMessageBox()
        # message_box.setText("Saving image is not implemented for this module.")
        # message_box.exec()
        QtGui.QMessageBox.information(self._widget, "Save image",
                                      "Saving image is not implemented for this module.")

class Controll(QtCore.QObject):
    """This class contains a widget for the user input. It also contains the Data and View
    and the communication between them."""
    def __init__(self, common_controll, viewer, data):
        super(Controll, self).__init__()
        self._common_controll = common_controll
        self._viewer = viewer
        self._data = data
        self._widget = QtGui.QWidget()
        self._active = True

    def initialize(self):
        """Any initialization that should be done after the viewer is
        properly set up with a window to draw into."""
        pass

    def get_viewer(self):
        """Returns the Viewer object."""
        return self._viewer

    def get_data(self):
        """Returns the Data object."""
        return self._data

    def get_widget(self):
        """Returns the widget containing the controlls."""
        return self._widget

    def set_active(self, state):
        """Tells the class that this is now the visible widget."""
        self._active = bool(state)

    def draw_hard(self):
        """Draw the widget."""
        pass

    def draw(self):
        """Draws the widget (using draw_hard()) if the current module is active"""
        if self._active:
            self.draw_hard()

    def on_dir_change(self):
        """Reload data andr draw when the directory changes."""
        self.draw()

class Plugin(QtCore.QObject):
    """Collects all classes of the module.
    Somewhat redundant with Controll but required to make it easier to reuse parts of
    a module in another (see for example weightmap_module."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(parent)
        self._common_controll = common_controll
        self._viewer = None
        self._data = None
        self._controll = None

    def get_viewer(self):
        """Retruns the Viewer object."""
        return self._viewer

    def get_data(self):
        """Retruns the Data object."""
        return self._data

    def get_controll(self):
        """Retruns the Controll object."""
        return self._controll

    def get_common_controll(self):
        """Retruns the CommonControll object. This is a global controll for
        module independent things, implemented in viewer.py"""
        return self._common_controll

    def initialize(self):
        self.get_viewer().initialize()
        self.get_data().initialize()
        self.get_controll().initialize()

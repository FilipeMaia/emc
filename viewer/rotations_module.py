"""Plot the distribution of responsabilities in rotational space.
In plane rotations are excluded to make the data 2D."""
import numpy
import h5py
from functools import partial
import rotations
import icosahedral_sphere
import module_template
from QtVersions import QtCore, QtGui
import vtk
import vtk_tools
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import convenient_widgets
import os

def enum(*enums):
    """Gives enumerate functionality."""
    return type('Enum', (), dict(zip(enums, range(len(enums)))))

ROTATION_TYPE = enum('single', 'average')

class RotationData(module_template.Data):
    """Reads data as requested and projects it to a sphere."""
    def __init__(self):
        super(RotationData, self).__init__()
        self._rotation_type = ROTATION_TYPE.single # 0 = average, 1 = single
        self._setup_done = False
        self._number_of_rotations = None
        self._number_of_bins = None
        self._rotation_sphere_weights = None
        self._rotation_sphere_coordinates = None
        self._rotation_sphere_good_indices = None
        self._rotation_mapping_table = None
        self._rotation_sphere_bins = None
        self._rotations_n = 10

        #self._setup_rotations()

    @staticmethod
    def _read_average_resp(iteration):
        """Read the responsabilities averaged for all images
        (these are precalculated in EMC)."""
        if os.path.isfile("output/average_resp_%.4d.h5" % (iteration)):
            with h5py.File("output/average_resp_%.4d.h5" % (iteration), "r") as file_handle:
                average_resp = file_handle["data"][...]
        else:
            average_resp = numpy.loadtxt("output/average_resp_%.4d.data" % (iteration))
        return average_resp

    def set_sampling_coordinates(self, coordinates):
        self._rotation_sphere_coordinates = coordinates

    def setup_rotations(self):
        """Read the appropriate quaternion list and calculate a table for projecting."""
        self._number_of_rotations = len(self._read_average_resp(0))
        self._rotations_n = rotations.rots_to_n(self._number_of_rotations)

        if self._rotation_sphere_coordinates is None:
            raise RuntimeError("Must call set_sampling_coordinates before setting up rotations")

        filename = "{directory}/data/rotations_table_n{n}.h5".format(directory=os.path.split(__file__)[0], n=self._rotations_n)
        print filename
        with h5py.File(filename, "r") as file_handle:
            self._rotation_sphere_weights = file_handle['weights'][...]
            self._rotation_mapping_table = file_handle['table'][...]

        self._rotation_sphere_good_indices = self._rotation_sphere_weights > 0.

        self._number_of_bins = len(self._rotation_sphere_weights)
        self._rotation_sphere_bins = numpy.zeros(self._number_of_bins)

        self._setup_done = True

    def is_ready(self):
        return self._setup_done

    def check_ready(self):
        if not self.is_ready():
            raise ValueError("RotationData is not set up. Call setup_rotations first.")

    def get_sampling_n(self):
        return self._rotations_n

    def get_rotation_coordinates(self):
        """Get the coordinates of the points of the sphere."""
        self.check_ready()
        # best_index = list(int32(loadtxt('output/best_rot.data')[self._current_iteration]))
        # return self._rotation_sphere_coordinates
        return self._rotation_sphere_coordinates

    def get_average_rotation_values(self, iteration):
        """Get the average rotation distribution of all images"""
        self.check_ready()
        try:
            #average_resp = loadtxt('output/average_resp_%.4d.data' % iteration)
            average_resp = self._read_average_resp(iteration)
        except IOError:
            self.read_error.emit()
            return self._rotation_sphere_bins
        if len(average_resp) != self._number_of_rotations:
            self._number_of_rotations = len(average_resp)
            self._rotations_n = rotations.rots_to_n(self._number_of_rotations)
            self.properties_changed.emit()
        self._rotation_sphere_bins[:] = 0.
        for i, resp in enumerate(average_resp):
            self._rotation_sphere_bins[self._rotation_mapping_table[i]] += resp
        self._rotation_sphere_bins[self._rotation_sphere_good_indices] /= \
            self._rotation_sphere_weights[self._rotation_sphere_good_indices]
        #return float32(self._rotation_sphere_weights) # debug line
        return self._rotation_sphere_bins

    def get_single_rotation_values(self, iteration, image_number):
        """Get the rotation distribution of a single image"""
        self.check_ready()
        try:
            resp_handle = h5py.File('output/responsabilities_%.4d.h5' % iteration)
            resp = resp_handle['data'][image_number, :]
            resp_handle.close()
        except IOError:
            self.read_error.emit()
        if len(resp) != self._number_of_rotations:
            self._number_of_rotations = len(resp)
            self._rotations_n = rotations.rots_to_n(self._number_of_rotations)
            self.properties_changed.emit()
        self._rotation_sphere_bins[:] = 0.
        for i, resp in enumerate(resp):
            self._rotation_sphere_bins[self._rotation_mapping_table[i]] += resp
        self._rotation_sphere_bins[self._rotation_sphere_good_indices] /= \
            self._rotation_sphere_weights[self._rotation_sphere_good_indices]
        return self._rotation_sphere_bins

class RotationViewer(module_template.Viewer):
    """Uses mayavi to plot a sphere made up of small spheres as pixels. The
    colors indicate the responsability density."""
    def __init__(self, parent=None):
        super(RotationViewer, self).__init__()
        #self._widget = QtGui.QWidget(parent)
        self._vtk_widget = QVTKRenderWindowInteractor(self._widget) # _vtk_widget is actually the RenderWindowInteractor
        self._vtk_widget.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())

        #self._vtk_widget.Initialize()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._vtk_widget)
        self._widget.setLayout(layout)

        self._renderer = vtk.vtkRenderer()
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)

        #self._mlab_widget = embedded_mayavi.MlabWidget()
        self._setup_done = False
        self._points = None
        self._default_sphere_n = 10
        self._sphere = vtk_tools.SphereMap(self._default_sphere_n)
        self._renderer.AddViewProp(self._sphere.get_actor())
        self._renderer.SetBackground(1., 1., 1.)
        #self._renderer.Render()

    def initialize(self):
        self._vtk_widget.Initialize()

    # def get_widget(self):
    #     """Return the widget containing the view."""
    #     return self._vtk_widget

    def set_sampling_n(self, sampling_n):
        """Rerun when the array size changes."""
        self._sphere.set_n(sampling_n)

    def get_coordinates(self):
        return self._sphere.get_coordinates()

    def get_number_of_points():
        pass

    def plot_rotations(self, values):
        """Update the viewer to show the new values."""
        if len(values) != self._sphere.get_number_of_points():
            #raise ValueError("values must be array of length {0}. Length {1} array received.".format(self._sphere.get_number_of_points(), len(values)))
            sampling_n = icosahedral_sphere.points_to_n(len(values))
            self.set_sampling_n(sampling_n)
        self._sphere.set_lookup_table(vtk_tools.get_lookup_table(0., values.max(), log=False, colorscale="jet", number_of_colors=1000))
        self._sphere.set_values(values)
        #self._sphere
        #self._renderer.Render()
        self._vtk_widget.Render()

class RotationControll(module_template.Controll):
    """Provides a widget for controlling the module and handles the calls to get the
    data and viewer classes."""
    class State(object):
        """This class contains the current state of the plot to separate them from the rest of the class"""
        def __init__(self):
            self.image_number = 0
            self.rotation_type = ROTATION_TYPE.average

    def __init__(self, common_controll, viewer, data):
        super(RotationControll, self).__init__(common_controll, viewer, data)

        self._show_average_rotation_radio = None
        self._show_single_rotation_radio = None
        self._rotation_image_number_box = None
        self._single_slice_widget = None

        self._state = self.State()
        self._setup_gui()
        self._setup_rotation_view()
        self._data.properties_changed.connect(self._setup_rotation_view)

    def _setup_rotation_view(self):
        """Initialize the viewer."""
        self._viewer.set_sampling_n(self._data.get_sampling_n())
        self._data.set_sampling_coordinates(self._viewer.get_coordinates())
        self._data.setup_rotations()
        #self._viewer.plot_rotations_init(self._data.get_rotation_coordinates())

    def _setup_gui(self):
        """Create the gui for the widget."""
        self._show_average_rotation_radio = QtGui.QRadioButton("Average rot")
        self._show_average_rotation_radio.setChecked(True)
        self._show_single_rotation_radio = QtGui.QRadioButton("Single rot")

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._show_average_rotation_radio)
        layout.addWidget(self._show_single_rotation_radio)
        self._show_average_rotation_radio.clicked.connect(partial(self.set_rotation_type,
                                                                  rotation_type=ROTATION_TYPE.average))
        self._show_single_rotation_radio.clicked.connect(partial(self.set_rotation_type,
                                                                 rotation_type=ROTATION_TYPE.single))

        self._rotation_image_number_box = QtGui.QSpinBox()
        self._rotation_image_number_box.setRange(-1, 10000)

        self._single_slice_widget = convenient_widgets.IntegerControll(0)
        self._single_slice_widget.valueChanged.connect(self.set_image_number)

        layout.addWidget(self._single_slice_widget)

        self._widget.setLayout(layout)
        self._single_slice_widget.hide()

    def draw_hard(self):
        """Draw the scene. Don't call this function directly. The draw() function calls this one
        if the module is visible."""
        if self._state.rotation_type == ROTATION_TYPE.average:
            self._viewer.plot_rotations(self._data.get_average_rotation_values(
                self._common_controll.get_iteration()))
            #self._rotation_image_number_box.hide()
            self._single_slice_widget.hide()
        elif self._state.rotation_type == ROTATION_TYPE.single:
            try:
                self._viewer.plot_rotations(self._data.get_single_rotation_values(
                    self._common_controll.get_iteration(), self._state.image_number))
            except IOError:
                print "Problem reading"
            #self._rotation_image_number_box.show()
            self._single_slice_widget.show()

    def set_rotation_type(self, rotation_type):
        """Choose between plotting responsabilities for a single image or
        an average over all images."""
        self._state.rotation_type = rotation_type
        self.draw()

    def set_image_number(self, new_image_number):
        """Set the image used for plotting in single image mode."""
        if new_image_number >= 0:
            self._state.image_number = new_image_number
            self.draw()

class Plugin(module_template.Plugin):
    """Collects all parts of the plugin."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._viewer = RotationViewer()
        self._data = RotationData()
        self._controll = RotationControll(self._common_controll, self._viewer, self._data)

"""Module for emc viewer. Used for plotting the assembled model."""
import numpy
import sphelper
import module_template
from QtVersions import QtCore, QtGui
import vtk
import vtk_tools
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from functools import partial
import enum
#import view_3d

INIT_SURFACE_LEVEL = 0.5

# def enum(*enums):
#     """Gives enumerate functionality."""
#     return type('Enum', (), dict(zip(enums, range(len(enums)))))

# VIEW_TYPE = enum('surface', 'slice')
VIEW_TYPE = enum.Enum("View type", ["surface", "slice"])

class ModelmapData(module_template.Data):
    """Reads data as requested."""
    def __init__(self, file_prefix):
        super(ModelmapData, self).__init__()
        self._file_prefix = file_prefix
        self._side = None

    def get_map(self, iteration):
        """Return the 3D model as an array for the requested iteration."""
        try:
            if iteration >= 0:
                modelmap = sphelper.import_spimage('%s_%.4d.h5' % (self._file_prefix, iteration), ['image'])
            elif iteration == -1:
                modelmap = sphelper.import_spimage('%s_init.h5' % (self._file_prefix), ['image'])
            elif iteration == -2:
                modelmap = sphelper.import_spimage('%s_final.h5' % (self._file_prefix), ['image'])
        except (IOError, KeyError):
            self.read_error.emit()
            if self._side:
                return numpy.zeros((self._side, )*3)
            else:
                return numpy.zeros((10, )*3)
        if self._side != modelmap.shape[0]:
            self._side = modelmap.shape[0]
            self.properties_changed.emit()
        return modelmap

    def get_mask(self, iteration):
        """Return the mask of the 3D model as an array for the requested iteration."""
        try:
            if iteration >= 0:
                mask = sphelper.import_spimage('%s_%.4d.h5' % (self._file_prefix, iteration), ['mask'])
            elif iteration == -1:
                mask = sphelper.import_spimage('%s_init.h5' % (self._file_prefix), ['mask'])
            elif iteration == -2:
                mask = sphelper.import_spimage('%s_final.h5' % (self._file_prefix), ['mask'])
        except IOError:
            self.read_error.emit()
            return
        if self._side != mask.shape[0]:
            self._side = mask.shape[0]
            self.properties_changed.emit()
        return mask

    def get_side(self):
        """Return the number fo pixels in the side of the 3D model side.
        (The model is always cubic)"""
        return self._side

class ModelmapViewer(module_template.Viewer):
    """Uses vtk to display the model in 3D as an isosurface or a slice. This is
    not a widget but contains one, accessible with get_widget()"""
    def __init__(self, parent=None):
        super(ModelmapViewer, self).__init__(parent)

        self._surface_algorithm = None
        self._surface_actor = None
        self._volume_scalars = None
        self._volume = None
        self._surface_level = 0.5
        self._color = (0.2, 0.8, 0.2)
        self._planes = []
        self._volume_max = 0.
        self._volume_numpy = None

        self._vtk_widget = QVTKRenderWindowInteractor(self._widget) # _vtk_widget is actually the RenderWindowInteractor
        self._vtk_widget.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._vtk_widget)

        self._widget.setLayout(layout)
        #self._vtk_widget.Initialize()
        #self._vtk_widget.Start()

        self._renderer = vtk.vtkRenderer()
        self._vtk_render_window = self._vtk_widget.GetRenderWindow()
        self._vtk_render_window.AddRenderer(self._renderer)
        #self._renderer = self._vtk_widget.GetRenderWindow().GetRenderer()
        self._lut = vtk_tools.get_lookup_table(0., 1., log=False, colorscale="jet")

        self._create_volume_map()

        white = (1., 1., 1.)
        self._renderer.SetBackground(white)
        #self.set_view_type(VIEW_TYPE.slice)
        #self._vtk_widget.GetRenderWindow().Render()

    def initialize(self):
        self._vtk_widget.Initialize()
        self._setup_slices()
        self._setup_surface()
        #self.set_view_type(VIEW_TYPE.surface)
        self.set_view_type(VIEW_TYPE.slice)
        #self._vtk_widget.GetRenderWindow().Render()
        self._renderer.ResetCamera()
        # camera = self._renderer.GetActiveCamera()
        # camera.SetPosition(2., 2., 2.)
        # camera.SetFocalPoint(0., 0., 0.)
        self._vtk_render_window.Render()
        # print self._renderer.GetVolumes()
        # print self._renderer.VisibleActorCount()
        # print self._surface_actor.GetBounds()
        # print self._renderer.GetActiveCamera().GetPosition()

    def _create_volume_map(self):
        """Create the vtk objects containing the data."""
        self._volume_max = 1.

        self._volume = vtk.vtkImageData()
        default_side = 100
        self._volume.SetExtent(0, default_side-1, 0, default_side-1, 0, default_side-1)

        self._volume_numpy = numpy.zeros((default_side, )*3, dtype="float32", order="C")
        self._volume_numpy[:] = 0.
        self._volume_max = self._volume_numpy.max()
        self._volume_scalars = vtk.vtkFloatArray()
        self._volume_scalars.SetNumberOfValues(default_side**3)
        self._volume_scalars.SetNumberOfComponents(1)
        self._volume_scalars.SetName("Values")
        self._volume_scalars.SetVoidArray(self._volume_numpy, default_side**3, 1)

        self._volume.GetPointData().SetScalars(self._volume_scalars)

    def _setup_surface(self):
        """Create the isosurface object, mapper and actor"""
        self._surface_level = INIT_SURFACE_LEVEL
        self._surface_algorithm = vtk.vtkMarchingCubes()
        self._surface_algorithm.SetInputData(self._volume)
        self._surface_algorithm.ComputeNormalsOn()
        self._surface_algorithm.SetValue(0, self._surface_level)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._surface_algorithm.GetOutputPort())
        mapper.ScalarVisibilityOff()
        self._surface_actor = vtk.vtkActor()
        self._surface_actor.GetProperty().SetColor(self._color[0], self._color[1], self._color[2])
        self._surface_actor.SetMapper(mapper)

        self._renderer.AddViewProp(self._surface_actor)

    def _setup_slices(self):
        """Create the slices. No actor required in this case"""
        picker = vtk.vtkCellPicker()
        picker_tolerance = 0.005
        picker.SetTolerance(picker_tolerance)
        text_color = (0., 0., 0.)

        self._planes.append(vtk.vtkImagePlaneWidget())
        self._planes[0].SetInputData(self._volume)
        self._planes[0].UserControlledLookupTableOn()
        self._planes[0].SetLookupTable(self._lut)
        self._planes[0].SetPlaneOrientationToXAxes()
        self._planes[0].SetSliceIndex(self._volume.GetExtent()[1]/2) # GetExtent returns a six length array, begin-end pairs
        self._planes[0].DisplayTextOn()
        self._planes[0].GetTextProperty().SetColor(text_color)
        self._planes[0].SetPicker(picker)
        self._planes[0].SetLeftButtonAction(1)
        self._planes[0].SetMiddleButtonAction(2)
        self._planes[0].SetRightButtonAction(0)
        self._planes[0].SetInteractor(self._vtk_widget)
        # self._planes[0].On()

        self._planes.append(vtk.vtkImagePlaneWidget())
        self._planes[1].SetInputData(self._volume)
        self._planes[1].UserControlledLookupTableOn()
        self._planes[1].SetLookupTable(self._lut)
        self._planes[1].SetPlaneOrientationToZAxes()
        self._planes[1].SetSliceIndex(self._volume.GetExtent()[5]/2) # GetExtent returns a six length array, begin-end pairs
        self._planes[1].DisplayTextOn()
        self._planes[1].GetTextProperty().SetColor(text_color)
        self._planes[1].SetPicker(picker)
        self._planes[1].SetLeftButtonAction(1)
        self._planes[1].SetMiddleButtonAction(2)
        self._planes[1].SetRightButtonAction(0)
        self._planes[1].SetInteractor(self._vtk_widget)
        # self._planes[1].On()

    def set_surface_visibility(self, state):
        """Hide or show the surface, used when swithching between the surface and"""
        if state:
            self._surface_actor.SetVisibility(1)
        else:
            self._surface_actor.SetVisibility(0)

    def set_slice_visibility(self, state):
        """Hide or show the slices, used when swithching between the surface and"""
        if state:
            self._planes[0].SetEnabled(1)
            self._planes[1].SetEnabled(1)
        else:
            self._planes[0].SetEnabled(0)
            self._planes[1].SetEnabled(0)

    def set_view_type(self, view_type):
        """Switch between viewing slices or isosurface."""
        if view_type == VIEW_TYPE.surface:
            self.set_slice_visibility(False)
            self.set_surface_visibility(True)
        elif view_type == VIEW_TYPE.slice:
            self.set_surface_visibility(False)
            self.set_slice_visibility(True)

    def _update_surface_level(self):
        """Set the isosurface level based on the relative level in _surface_level
        and the maximum value of the current model."""
        self._surface_algorithm.SetValue(0, self._surface_level*self._volume_max)
        self._surface_algorithm.Modified()
        self._vtk_widget.Render()

    def set_surface_level(self, value):
        """Change the relative isosurface level"""
        if value < 0. or value > 1.:
            raise ValueError("Surface value must be in [0.,1.], was %g\n", value)
        self._surface_level = value
        self._update_surface_level()

    def get_surface_level(self):
        """Get the relative isosurface level"""
        return self._surface_level

    def plot_map(self, new_data_array):
        """Update the viwer to show the provided map."""
        self._volume_numpy[:, :, :] = new_data_array
        self._volume_max = new_data_array.max()
        self._lut.SetTableRange(new_data_array.min(), new_data_array.max())
        self._update_surface_level()
        self._volume_scalars.Modified()
        #self._vtk_widget.GetRenderWindow().Render()
        self._vtk_render_window.Render()

    def plot_map_init(self, new_data_array):
        """As opposed to plot_map() this function accepts maps of different side than
        the active one"""
        self._volume_max = new_data_array.max()

        old_extent = numpy.array(self._volume.GetExtent())

        self._volume.SetExtent(0, new_data_array.shape[0]-1, 0, new_data_array.shape[1]-1, 0, new_data_array.shape[2]-1)

        self._volume_numpy = numpy.ascontiguousarray(new_data_array, dtype="float32")

        self._volume_scalars = vtk.vtkFloatArray()
        self._volume_scalars.SetNumberOfValues(numpy.product(self._volume_numpy.shape))
        self._volume_scalars.SetNumberOfComponents(1)
        self._volume_scalars.SetName("Values")
        self._volume_scalars.SetVoidArray(self._volume_numpy, numpy.product(self._volume_numpy.shape), 1)

        self._volume.GetPointData().SetScalars(self._volume_scalars)

        # self._volume_scalars.SetNumberOfValues(new_data_array.shape[0]*new_data_array.shape[1]*new_data_array.shape[2])

        # for i, this_value in enumerate(numpy.ravel(new_data_array.swapaxes(0, 2))):
        #     self._volume_scalars.SetValue(i, this_value)

        self._lut.SetTableRange(new_data_array.min(), new_data_array.max())
        #self._volume.GetPointData().SetScalars(self._volume_scalars)
        self._update_surface_level()
        self._volume_scalars.Modified()
        self._volume.Modified()
        new_extent = numpy.array(self._volume.GetExtent())
        scaling_factors = numpy.float64(new_extent[1::2]) / numpy.float64(old_extent[1::2])
        self._planes[0].SetOrigin(numpy.array(self._planes[0].GetOrigin()*scaling_factors))
        self._planes[0].SetPoint1(numpy.array(self._planes[0].GetPoint1()*scaling_factors))
        self._planes[0].SetPoint2(numpy.array(self._planes[0].GetPoint2()*scaling_factors))
        self._planes[1].SetOrigin(numpy.array(self._planes[1].GetOrigin()*scaling_factors))
        self._planes[1].SetPoint1(numpy.array(self._planes[1].GetPoint1()*scaling_factors))
        self._planes[1].SetPoint2(numpy.array(self._planes[1].GetPoint2()*scaling_factors))
        # self._planes[0].Modified()
        # self._planes[1].Modified()

        self._planes[0].UpdatePlacement()
        self._planes[1].UpdatePlacement()
        self._vtk_widget.Render()
        self._renderer.Render()


class ModelmapControll(module_template.Controll):
    """Provides a widget for controlling the module and handles the calls to get the
    data and viewer classes."""
    # class State(object):
    #     """This class contains the current state of the plot to separate them from the rest of the class"""
    #     def __init__(self):
    #         self.view_type = VIEW_TYPE.surface
    #         self.log_scale = False

    def __init__(self, common_controll, viewer, data):
        super(ModelmapControll, self).__init__(common_controll, viewer, data)
        self._slider_length = 1000
        self._surface_level_slider = None
        self._log_scale_widget = None

        #self._state = self.State()
        self._state = {"view_type": VIEW_TYPE.surface,
                       "log_scale": False}
        #self._viewer.plot_map_init(self._data.get_map(self._common_controll.get_iteration()))
        self._setup_gui()
        self._data.properties_changed.connect(self._setup_viewer)

    def initialize(self):
        self._viewer.plot_map_init(self._data.get_map(self._common_controll.get_iteration()))
        self._surface_level_slider.blockSignals(False)

    def _setup_viewer(self):
        """Provedes an empty map to the viewer to initialize it."""
        self._viewer.plot_map_init(numpy.zeros((self._data.get_side(), )*3))

    def _setup_gui(self):
        """Create the gui for the widget."""
        view_type_radio_surface = QtGui.QRadioButton("Isosurface plot")
        view_type_radio_slice = QtGui.QRadioButton("Slice plot")
        view_type_radio_slice.setChecked(True)
        view_type_layout = QtGui.QVBoxLayout()
        view_type_layout.addWidget(view_type_radio_surface)
        view_type_layout.addWidget(view_type_radio_slice)
        view_type_radio_surface.clicked.connect(partial(self.set_view_type, VIEW_TYPE.surface))
        view_type_radio_slice.clicked.connect(partial(self.set_view_type, VIEW_TYPE.slice))

        #surface controll setup
        self._surface_level_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self._surface_level_slider.setTracking(True)
        self._surface_level_slider.setRange(1, self._slider_length)
        def on_slider_changed(self, slider_level):
            """Handles the signal from a surface level slider change."""
            # surface_level = slider_level/float(cls._slider_length)
            # surface_level = self._slider_length/(float(slider_level))
            surface_level = ((numpy.exp(float(slider_level)/float(self._slider_length))-1.) /
                             (numpy.exp(1.)-1.))
            self._viewer.set_surface_level(surface_level)
        self._surface_level_slider.valueChanged.connect(partial(on_slider_changed, self))
        self._surface_level_slider.blockSignals(True)
        self._surface_level_slider.setSliderPosition(self._slider_length*INIT_SURFACE_LEVEL)

        # slice controll widget setup
        log_scale_box = QtGui.QCheckBox()
        log_scale_label = QtGui.QLabel('Log Scale')
        log_scale_layout = QtGui.QHBoxLayout()
        log_scale_layout.addWidget(log_scale_label)
        log_scale_layout.addWidget(log_scale_box)
        log_scale_layout.addStretch()
        log_scale_box.stateChanged.connect(self.set_log_scale)
        self._log_scale_widget = QtGui.QWidget()
        self._log_scale_widget.setLayout(log_scale_layout)
        #self._log_scale_widget.hide()

        layout = QtGui.QVBoxLayout()
        layout.addLayout(view_type_layout)
        layout.addWidget(self._surface_level_slider)
        layout.addWidget(self._log_scale_widget)
        #layout.addStretch()
        self._widget.setLayout(layout)
        self._widget.setFixedHeight(120)

    def draw_hard(self):
        """Draw the scene. Don't call this function directly. The draw() function calls this one
        if the module is visible."""
        if (self._state["view_type"] == VIEW_TYPE.slice) and self._state["log_scale"]:
            self._viewer.plot_map(numpy.log(1.+self._data.get_map(self._common_controll.get_iteration())))
        else:
            self._viewer.plot_map(self._data.get_map(self._common_controll.get_iteration()))

    def set_view_type(self, view_type):
        """Select between isosurface and slice plot."""
        self._state["view_type"] = view_type
        self._viewer.set_view_type(view_type)
        if view_type == VIEW_TYPE.surface:
            self._log_scale_widget.hide()
            self._surface_level_slider.show()
        elif view_type == VIEW_TYPE.slice:
            self._surface_level_slider.hide()
            self._log_scale_widget.show()
        self.draw()

    def set_log_scale(self, state):
        """Set wether log scale is used for the slice plot."""
        self._state["log_scale"] = bool(state)
        self.draw()


class Plugin(module_template.Plugin):
    """Collects all parts of the plugin."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll, parent)
        self._viewer = ModelmapViewer()
        self._data = ModelmapData("output/model")
        self._controll = ModelmapControll(self._common_controll, self._viewer, self._data)

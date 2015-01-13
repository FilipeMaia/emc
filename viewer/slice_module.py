"""
Module to plot diffraction slices

We generally use the indexing [x,y] and [x,y,z].
"""
import numpy
from PyQt4 import QtCore, QtGui
import vtk
import h5py
import module_template
import os
import re
import sphelper
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import rotations
import vtk_tools

class SliceData(module_template.Data):
    """Provides the diffraction pattern data and the rotations."""
    def __init__(self):
        super(SliceData, self).__init__()
        self._rotations = None
        #self._images = []
        self._images = {}
        #self._masks = []
        self._masks = {}
        self._iteration = 0
        self._number_of_images = None
        self._scaling_file_handle = None #fix this soon
        self._open_scaling_file()

        # self.dir_changed()

    def initialize(self):
        self.dir_changed()

    def __del__(self):
        self._close_scaling_file()

    def _read_rotations(self):
        """Read rotations and save them within the object."""
        self._rotations = numpy.loadtxt("output/best_quaternion_%.4d.data" % (self._iteration))

    def _set_number_of_images(self):
        prefix = "output"
        list_of_all_files = os.listdir(prefix)
        list_of_images = [f for f in list_of_all_files if re.search("^image_[0-9]{4}.h5$", f)]
        self._number_of_images = len(list_of_images)

    def _read_images(self):
        """Read diffraction patterns and save them within the object."""
        #should I normalize images??
        list_of_all_files = os.listdir("output")
        list_of_images = [f for f in list_of_all_files if re.search("^image_[0-9]{4}.h5$", f)]
        prefix = "output"
        if len(list_of_images) == 0:
            # maybe this is an old run with images output in debug
            list_of_all_files = os.listdir("debug")
            list_of_images = [f for f in list_of_all_files if re.search("^image_[0-9]{4}.h5$", f)]
            prefix = "debug"
        if len(list_of_images) == 0:
            self.read_error.emit()
        self._number_of_images = len(list_of_images)
        for i in range(self._number_of_images):
            try:
                image, mask = sphelper.import_spimage("%s/image_%.4d.h5" % (prefix, i), ['image', 'mask'])
            except IOError:
                print "{0} is bad".format(i)
                image = numpy.zeros(self._images[0].shape, dtype="float32")
                mask = numpy.zeros(self._images[0].shape, dtype="int32")
            self._images.append(image)
            self._masks.append(mask)

    def _open_scaling_file(self):
        """Open scaling file and keep a handle for use by the object."""
        try:
            self._scaling_file_handle = h5py.File("output/scaling_%.4d.h5" % self._iteration, "r")
        except IOError:
            self._scaling_file_handle = None

    def _close_scaling_file(self):
        """Close the scaling file handle."""
        if self._scaling_file_handle:
            self._scaling_file_handle.close()

    def _update_scaling_handle(self):
        """Close and open file, use for example on directory changes."""
        self._close_scaling_file()
        self._open_scaling_file()

    # def dir_changed(self):
    #     """Reload both diffraction patterns and rotations."""
    #     self._read_images()
    #     self._read_rotations()
    #     self._update_scaling_handle()

    def dir_changed(self):
        """Signal that the currently stored images and rotations are outdated."""
        self._images.clear()
        self._masks.clear()
        self._set_number_of_images()
        self._read_image_and_mask(0)
        self._read_rotations()
        self._update_scaling_handle()

    def set_iteration(self, new_iteration):
        """Change iteration and update rotations accordingly."""
        if new_iteration >= 0:
            self._iteration = new_iteration
            self._read_rotations()

    # def get_image(self, image_index, slice_index=None):
    #     """Returns the diffraction pattern with the specified index. If a slice index is provided
    #     the data is rescaled with the appropriate scaling factor."""
    #     if slice_index == None:
    #         return self._images[image_index]
    #     else:
    #         return self._images[image_index] / self.get_scaling(image_index, slice_index)

    def _read_image_and_mask(self, index):
        """Load the image and mask if they have not been viewed before."""
        if index in self._images and index in self._masks:
            return
        prefix = "output"
        try:
            image, mask = sphelper.import_spimage(os.path.join(prefix, "image_{0:04}.h5".format(index)), ["image", "mask"])
        except IOError:
            print "Image {0} is bad".format(index)
            self._images[index] = numpy.zeros(self._images[0].shape, dtype="float64")
            self._masks[index] = numpy.zeros(self._images[0].shape, dtype="int32")
            return
        self._images[index] = image
        self._masks[index] = mask

    def get_image(self, image_index, slice_index=None, mask=True):
        self._read_image_and_mask(image_index)
        image = self._images[image_index]
        if slice_index:
            image = image / self.get_scaling(image_index, slice_index)
        if mask:
            image = image * self._masks[image_index]
        return image

    # def get_mask(self, index):
    #     """Returns the mask of the diffraction pattern."""
    #     return self._masks[index]

    def get_scaling(self, image_index, slice_index):
        """Returns the scaling of the image at the provided rotation (index)."""
        if self._scaling_file_handle:
            return self._scaling_file_handle['data'][slice_index, image_index]

    def get_rotation(self, index):
        """Get the rotation for the image with the provided index"""
        return self._rotations[index]

    @staticmethod
    def get_curvature():
        """Get the radius of the Ewald sphere of the current data in units of pixels."""
        return numpy.inf

    def get_number_of_images(self):
        """Returns the number of diffraction patterns."""
        return self._number_of_images

    def get_image_side(self):
        """Returns the image side in pixels."""
        self._read_image_and_mask(0)
        return self._images[0].shape[0]

    def get_total_max(self):
        """Gives an estimate of the maximum intensity in the entire dataset"""
        #this doesn't take scaling into account because it would take to much time
        return numpy.array([image.max() for image in self._images.values()]).max()

    def get_total_min(self):
        """Gives an estimate of the minimum intensity in the entire dataset"""
        #this doesn't take scaling into account because it would take to much time
        return numpy.array([image.min() for image in self._images.values()]).min()

class SliceGenerator(object):
    """Objects of this class creates vtk polydata objects from diffraction images and rotations."""
    def __init__(self, side, radius):
        self._side = side
        self._radius = radius
        x_array_single = numpy.arange(self._side) - self._side/2. + 0.5
        y_array_single = numpy.arange(self._side) - self._side/2. + 0.5
        self._x_array, self._y_array = numpy.meshgrid(x_array_single, y_array_single)
        if numpy.isinf(radius):
            self._z_array = numpy.zeros(self._x_array.shape)
        else:
            self._z_array = self._radius - numpy.sqrt(self._radius**2 - self._x_array**2 - self._y_array**2)

        self._image_values = vtk.vtkFloatArray()
        self._image_values.SetNumberOfComponents(1)
        self._image_values.SetName("Intensity")

        self._points = vtk.vtkPoints()
        #point_indices = -numpy.ones((self._side)*(self._side), dtype='int32')

        for i in range(self._side):
            for j in range(self._side):
                #self._points.InsertNextPoint(self._x_array[i, j], self._y_array[i, j], self._z_array[i, j])
                self._points.InsertNextPoint(self._z_array[i, j], self._y_array[i, j], self._x_array[i, j])
                self._image_values.InsertNextTuple1(0.)

        self._template_poly_data = self._circular_slice()

    def _square_slice(self):
        """Return a polydata of a square slice."""
        polygons = vtk.vtkCellArray()
        for i in range(self._side-1):
            for j in range(self._side-1):
                corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(4)
                for index, corner in enumerate(corners):
                    polygon.GetPointIds().SetId(index, corner[0]*self._side+corner[1])
                polygons.InsertNextCell(polygon)

        template_poly_data = vtk.vtkPolyData()
        template_poly_data.SetPoints(self._points)
        template_poly_data.GetPointData().SetScalars(self._image_values)
        template_poly_data.SetPolys(polygons)
        return template_poly_data

    def _circular_slice(self):
        """Return a cellarray of a square slice."""
        polygons = vtk.vtkCellArray()
        for i in range(self._side-1):
            for j in range(self._side-1):
                corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
                radius = max([numpy.sqrt((c[0] - self._side/2. + 0.5)**2 + (c[1] - self._side/2. + 0.5)**2)
                              for c in corners])
                if radius < self._side/2.:
                    polygon = vtk.vtkPolygon()
                    polygon.GetPointIds().SetNumberOfIds(4)
                    for index, corner in enumerate(corners):
                        polygon.GetPointIds().SetId(index, corner[0]*self._side+corner[1])
                    polygons.InsertNextCell(polygon)

        template_poly_data = vtk.vtkPolyData()
        template_poly_data.SetPoints(self._points)
        template_poly_data.GetPointData().SetScalars(self._image_values)
        template_poly_data.SetPolys(polygons)
        return template_poly_data

    def _get_rotated_coordinates(self, rot):
        """Returns the template coordinates of the slice rotated by the given rotation."""
        z_array, y_array, x_array = rotations.rotate_array(rot, self._z_array.flatten(),
                                                           self._y_array.flatten(),
                                                           self._x_array.flatten())
        x_array = x_array.reshape((self._side, self._side))
        y_array = y_array.reshape((self._side, self._side))
        z_array = z_array.reshape((self._side, self._side))
        return z_array, y_array, x_array

    def get_slice(self, image, rotation):
        """Generate a polydata object of the provided image rotated by the
        provided rotation."""

        rotation_degrees = rotation.copy()
        rotation_degrees[0] = 2.*numpy.arccos(rotation[0])*180./numpy.pi
        transformation = vtk.vtkTransform()
        transformation.RotateWXYZ(rotation_degrees[0], rotation_degrees[1],
                                  rotation_degrees[2], rotation_degrees[3])
        transform_filter = vtk.vtkTransformFilter()
        input_poly_data = vtk.vtkPolyData()
        input_poly_data.DeepCopy(self._template_poly_data)
        # This deep copy stuff is needed since transform returns a
        # pointer to the same object but transformed (i think) but
        # it should be checked really.
        #input_poly_data = self._template_poly_data.
        #transform_filter.SetInputData(self._template_poly_data)
        transform_filter.SetInputData(input_poly_data)
        transform_filter.SetTransform(transformation)
        transform_filter.Update()
        this_poly_data = transform_filter.GetOutput()

        scalars = this_poly_data.GetPointData().GetScalars()
        for i in range(self._side):
            for j in range(self._side):
                # point_coord = this_poly_data.GetPoint(self._side*i + j)
                #if point_coord[0] > 0.:
                scalars.SetTuple1(i*self._side+j, image[i, j])
                #scalars.SetTuple1(i*self._side+j, image[j, i])
                # else:
                #     #polys.GetData().SetTuple4(i*self._side+j, 0., 0., 0., 0.)
                #     scalars.SetTuple1(i*self._side+j, nan)
        this_poly_data.Modified()
        return this_poly_data


class SliceViewer(module_template.Viewer):
    """Plots the slices using vtk."""
    def __init__(self, data, parent):
        super(SliceViewer, self).__init__()
        self._data = data
        self._lut = None
        self._actors = None
        self._camera = None

        #self._slice_generator = SliceGenerator(self._data.get_image_side(), self._data.get_curvature())
        #self._workspace = QtGui.QWorkspace()
        self._widget = QtGui.QWidget(parent)
        self._vtk_widget = QVTKRenderWindowInteractor(self._widget)
        self._vtk_widget.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())
        #self._vtk_widget = QtGui.QPushButton("Foo", parent=self._widget)
        layout = QtGui.QVBoxLayout()
        #layout.addWidget(self._workspace)
        layout.addWidget(self._vtk_widget)
        #self._workspace.addWindow(self._vtk_widget)
        self._widget.setLayout(layout)
        # self._vtk_widget.Initialize()
        # self._vtk_widget.Start()
        self._renderer = vtk.vtkRenderer()
        self._vtk_render_window = self._vtk_widget.GetRenderWindow()
        self._vtk_render_window.AddRenderer(self._renderer)
        #self._setup_slice_view()

    def initialize(self):
        #pass
        self._vtk_widget.Initialize()
        self._setup_slice_view()
        self._slice_generator = SliceGenerator(self._data.get_image_side(), self._data.get_curvature())

    def _draw(self):
        if self._vtk_render_window.IsDrawable():
            self._vtk_render_window.Render()

    def _setup_slice_view(self):
        """Setup background, camera and LUT."""
        self._renderer.SetBackground(1, 1, 1)

        # camera
        self._camera = vtk.vtkCamera()
        self._camera.SetPosition(130, 130, 170)
        self._camera.SetFocalPoint(0., 0., 0.)
        self._camera.SetViewUp(0, 0, 1)
        self._renderer.SetActiveCamera(self._camera)

        # self._lut = vtk_tools.get_lookup_table(self._data.get_total_min(), self._data.get_total_max(),
        #                                        log=True, colorscale="jet")
        self._lut = vtk_tools.get_lookup_table(0.1, 10., log=True, colorscale="jet")
        self._actors = {}

    def _update_lut(self):
        """Call after new images were added to update the LUT to include
        the entire range."""
        # self._lut = vtk_tools.get_lookup_table(self._data.get_total_min(), self._data.get_total_max(),
        #                                        log=True, colorscale="jet")
        self._lut.SetTableRange(self._data.get_total_min(), self._data.get_total_max())
        self._lut.Build()
        self._draw()
        

    def _add_poly_data(self, this_poly_data, identifier):
        """Add a polydata and give it an id by which it can be accessed later."""
        if identifier in self._actors:
            raise ValueError("Actor with identifier %d is already plotted" % identifier)
        mapper = vtk.vtkPolyDataMapper()
        #mapper.SetInput(this_poly_data)
        mapper.SetInputData(this_poly_data)
        mapper.SetScalarModeToUsePointData()
        mapper.UseLookupTableScalarRangeOn()
        self._update_lut()
        mapper.SetLookupTable(self._lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        #actor.GetProperty().SetOpacity(0.999999)
        self._actors[identifier] = actor

        self._renderer.AddActor(actor)
        #self._renderer.UseDepthPeelingOn()

    def _remove_poly_data(self, identifier):
        """Remove polydata with the specified id."""
        if not identifier in self._actors:
            raise ValueError("Trying to remove actor with id %d that doesn't exist." % identifier)
        self._renderer.RemoveActor(self._actors[identifier])
        del self._actors[identifier]

    def add_slice(self, index):
        """Add the image with the specified index to the view."""
        image = self._data.get_image(index)
        self._add_poly_data(self._slice_generator.get_slice(self._data.get_image(index),
                                                            self._data.get_rotation(index)), index)
        self._draw()

    def add_multiple_slices(self, index_list):
        """Add all the specified images to the view."""
        for index in index_list:
            self._add_poly_data(self._slice_generator.get_slice(self._data.get_image(index),
                                                                self._data.get_rotation(index)), index)
        self._draw()

    def remove_slice(self, index, render=True):
        """Remove the specified image from the view."""
        self._remove_poly_data(index)
        #self._vtk_widget.GetRenderWindow().Render()
        if render:
            self._draw()

    def remove_all_slices(self):
        """Clear the view."""
        for index in self.get_all_indices():
            self._remove_poly_data(index)

    def get_all_indices(self):
        """Get all ids currently known to the viewer."""
        return self._actors.keys()


class SliceControll(module_template.Controll):
    """This class contains a widget for the user input. It also contains the Data and View
    and some of the communication between them."""
    def __init__(self, common_controll, viewer, data):
        super(SliceControll, self).__init__(common_controll, viewer, data)
        self._pattern_list = None
        self._check_all_box = None
        self._setup_gui()

    def initialize(self):
        self.on_dir_change()

    def _setup_gui(self):
        """Create the gui for the widget."""
        self._widget = QtGui.QWidget()
        layout = QtGui.QHBoxLayout()

        #pattern list
        self._pattern_list = QtGui.QListWidget()
        self._pattern_list.itemChanged.connect(self._on_list_item_changed)

        #self._set_number_of_images(self._data.get_number_of_images())

        # for i in range(self._data.get_number_of_images()):
        #     this_item = QtGui.QListWidgetItem("Image %d" % i, self._pattern_list)
        #     this_item.setFlags(this_item.flags() | QtCore.Qt.ItemIsUserCheckable)
        #     this_item.setData(QtCore.Qt.CheckStateRole, QtCore.Qt.Unchecked)

        self._check_all_box = QtGui.QCheckBox()
        check_all_layout = QtGui.QHBoxLayout()
        check_all_layout.addWidget(self._check_all_box)
        check_all_layout.addWidget(QtGui.QLabel("Check all"))
        check_all_layout.addStretch()

        self._check_all_box.stateChanged.connect(self._on_check_all_changed)

        layout.addWidget(self._pattern_list)
        layout.addLayout(check_all_layout)
        self._widget.setLayout(layout)

    def _set_number_of_images(self, number_of_images):
        old_blocking_state = self._pattern_list.blockSignals(True)
        if number_of_images > self._pattern_list.count():
            for i in range(self._pattern_list.count(), number_of_images):
                this_item = QtGui.QListWidgetItem("Image {0}".format(i), self._pattern_list)
                this_item.setFlags(this_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                this_item.setData(QtCore.Qt.CheckStateRole, QtCore.Qt.Unchecked)
        elif number_of_images < self._pattern_list.count():
            for __ in range(number_of_images, self._pattern_list.count()):
                self._pattern_list.removeItemWidget(self._pattern_list.item(number_of_images))
        self._pattern_list.blockSignals(old_blocking_state)
            

    def _on_list_item_changed(self, item):
        """This function handles additions and removals of single slices."""
        index = self._pattern_list.indexFromItem(item).row()
        if item.checkState() == QtCore.Qt.Checked:
            self._viewer.add_slice(index)
        else:
            self._viewer.remove_slice(index)

    def _on_check_all_changed(self, state):
        """This function handles addition and removal of all slices at once."""
        self._active = False
        for index in range(self._pattern_list.count()):
            if state:
                self._pattern_list.item(index).setData(QtCore.Qt.CheckStateRole, QtCore.Qt.Checked)
            else:
                self._pattern_list.item(index).setData(QtCore.Qt.CheckStateRole, QtCore.Qt.Unchecked)
        self._active = True
        self.draw()

    def on_dir_change(self):
        self._data.dir_changed()
        self._set_number_of_images(self._data.get_number_of_images())
        self.draw()

    def draw_hard(self):
        """Draw the widget. In practice call draw() instead as it only draws when the
        module is active."""
        self._data.set_iteration(self._common_controll.get_iteration())
        indices = self._viewer.get_all_indices()
        self._viewer.remove_all_slices()
        self._viewer.add_multiple_slices(indices)

class Plugin(module_template.Plugin):
    """Collects all classes of the module."""
    def __init__(self, common_controll, parent=None):
        super(Plugin, self).__init__(common_controll)
        self._data = SliceData()
        self._viewer = SliceViewer(self._data, parent)
        self._controll = SliceControll(common_controll, self._viewer, self._data)

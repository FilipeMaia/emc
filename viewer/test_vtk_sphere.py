import vtk
import vtk_tools
import numpy

sphere = vtk_tools.SphereMap(20)

#sphere.set_values(sphere.get_coordinates()[:, 0])
sphere.set_values(numpy.random.random(len(sphere.get_coordinates())))

renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.2, 0.3)
renderer.AddViewProp(sphere.get_actor())
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())
interactor.SetRenderWindow(render_window)

# plane_1.SetInteractor(interactor)
# plane_1.On()
# plane_2.SetInteractor(interactor)
# plane_2.On()

# renderer.AddViewProp(outline_actor)

render_window.Render()
interactor.Start()

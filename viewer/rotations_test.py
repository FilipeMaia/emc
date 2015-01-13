import vtk
import numpy
import sphelper
import view_3d
import itertools

import icosahedral_sphere

def icosahedron_edges():
    coordinates = icosahedral_sphere.icosahedron_vertices()
    edges = []
    indices = []
    # for c1 in coordinates:
    #     for c2 in coordinates:
    for c1, c2 in itertools.combinations(enumerate(coordinates), 2):
        if ((c1[1] == c2[1]).sum() < 3) and (numpy.linalg.norm(c1[1] - c2[1]) < 3.):
            edges.append((c1[1], c2[1]))
            indices.append((c1[0], c2[0]))
    return edges, indices

def icosahedron_faces():
    coordinates = icosahedral_sphere.icosahedron_vertices()
    face_cutoff = 1.5
    faces = []
    indices = []
    for c1, c2, c3 in itertools.combinations(enumerate(coordinates), 3):
        if ((c1[1]==c2[1]).sum() < 3) and ((c1[1]==c3[1]).sum() < 3) and ((c2[1]==c3[1]).sum() < 3):
            center = (c1[1]+c2[1]+c3[1])/3.
            if numpy.linalg.norm(center-c1[1]) < face_cutoff and numpy.linalg.norm(center-c2[1]) < face_cutoff and numpy.linalg.norm(center-c3[1]) < face_cutoff:
                faces.append((c1[1], c2[1], c3[1]))
                indices.append((c1[0], c2[0], c3[0]))
    return faces, indices

def sphere_poly_data():
    rotations_n = 1
    coordinates = numpy.array(icosahedral_sphere.sphere_sampling(rotations_n))
    
    #create points object 
    points = vtk.vtkPoints()
    for c in coordinates:
        points.InsertNextPoint(c[0], c[1], c[2])

    #coordinates = icosahedral_sphere.icosahedron_vertices()
    base_coordinates = icosahedral_sphere.icosahedron_vertices()
    edges, edge_indices = icosahedron_edges()
    faces, face_indices = icosahedron_faces()

    edge_points = []

    for e in edges:
        origin = e[0]
        base = e[1]-e[0]
        for i in range(1, rotations_n):
            edge_points.append(origin + i/float(rotations_n)*base)

    def get_index(i, j):
        return int((rotations_n+1)*j + float(j)/2. - float(j)**2/2. + i)

    face_points = []
    print "start loop"
    print zip(faces, face_indices)
    for f, fi in zip(enumerate(faces), face_indices):
        base_index = f[0] * (((rotations_n+1)**2 + rotations_n)/2)
        print base_index
        for i in range(0, rotations_n):
            for j in range(0, rotations_n):
                if i+j < rotations_n:
                    face_indices.append((base_index + get_index(i, j),
                                          base_index + get_index(i, j+1),
                                          base_index + get_index(i+1, j)))

    # full_list = [numpy.array(c) for c in coordinates] + edge_points + face_points
    # normalized_list =[l/numpy.linalg.norm(l) for l in full_list]

    points = vtk.vtkPoints()
    for c in coordinates:
        points.InsertNextPoint(c[0], c[1], c[2])
    print "number of points = {0}".format(points.GetNumberOfPoints())

    polygons = vtk.vtkCellArray()
    for p in face_indices:
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(3)
        for i, pi in enumerate(p):
            polygon.GetPointIds().SetId(i, pi)
        polygons.InsertNextCell(polygon)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(polygons)
    return poly_data
    #find polygons by searching for nearby points

    #create polygons object


radius = 0.5

#coordinates = numpy.array(icosahedral_sphere.sphere_sampling(rotations_n))


# sphere_source =vtk.vtkSphereSource()
# sphere_source.SetCenter(0, 0, 0)
# sphere_source.SetRadius(0.5)
# sphere_source.SetPhiResolution(100)
# sphere_source.SetThetaResolution(100)
# sphere_source.Update()

poly_data = sphere_poly_data()
print "poly_data done"
# poly_data = sphere_source.GetOutput()
# point_data = poly_data.GetPointData()

scalars = vtk.vtkFloatArray()
scalars.SetNumberOfValues(poly_data.GetPoints().GetNumberOfPoints())
for i in range(scalars.GetNumberOfTuples()):
    scalars.SetValue(i, numpy.random.random())

poly_data.GetPointData().SetScalars(scalars)
poly_data.Modified()
print "scalars done"

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(poly_data)
print "mapper done"
actor = vtk.vtkActor()
actor.SetMapper(mapper)
print "actor done"

renderer = vtk.vtkRenderer()
renderer.SetBackground(1., 1., 1.)
print "renderer done"
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
print "render window done"
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())
interactor.SetRenderWindow(render_window)
print "interactor done"

renderer.AddActor(actor)
print "add actor"
render_window.Render()
print "render"
#interactor.Start()
print "start interactor"


import matplotlib.pyplot as plt
import matplotlib.tri as tri
from dolfin import *
import numpy as np

domain = Rectangle(-1, -1, 1, 1) - Circle(0, 0, 0.5)
mesh = Mesh(domain, 20)
n = mesh.num_vertices()
d = mesh.geometry().dim()

# Create the triangulation
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0],
                                  mesh_coordinates[:, 1],
                                  triangles)

# Plot the mesh
plt.figure()
plt.triplot(triangulation)
plt.savefig('mesh.png')

# Create some function
V = FunctionSpace(mesh, 'CG', 1)
f_exp = Expression('sin(2*pi*(x[0]*x[0]+x[1]*x[1]))')
f = interpolate(f_exp, V)

# Get the z values as face colors for each triangle(midpoint)
plt.figure()
zfaces = np.asarray([f(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.savefig('f0.png')

# Get the z values for each vertex
plt.figure()
z = np.asarray([f(point) for point in mesh_coordinates])
plt.tripcolor(triangulation, z, edgecolors='k')
plt.savefig('f1.png')

# Comment to prevent pop-up
plt.show()

# for ipython notebook
%matplotlib inline


def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def plot(obj):
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C)
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')

# example
mesh = UnitSquareMesh(10, 10)
plt.figure()
plot(mesh)
plt.show()
Q = FunctionSpace(mesh, "CG", 1)
F = interpolate(Expression("x[0]"), Q)
plot(F)
plt.show()

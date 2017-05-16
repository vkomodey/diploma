from dolfin import *
from fenicstools import interpolate_nonmatching_mesh

def get_whole_function(V, mesh,u1,u2,domains):
    u2 = interpolate_nonmatching_mesh ( u2 , V)
    V_dofmap = V.dofmap()
    chi1 = Function(V)
    chi2 = Function(V)
    gamma_dofs = []
    for cell in cells(mesh): # set the characteristic functions
        if domains[cell] == 1:
            chi1.vector()[V_dofmap.cell_dofs(cell.index())] = 1
            gamma_dofs.extend(V_dofmap.cell_dofs(cell.index()))
        else:
            chi2.vector()[V_dofmap.cell_dofs(cell.index())]=1
    gamma_dofs = list(set(gamma_dofs))
    u2.vector()[gamma_dofs] = 0
    u_0 = project(chi1*u1, V)
    u_0 += project(chi2*u2, V)
    return u_0

def define_domains (mesh, V, cells_num) :
    domains = CellFunction("size_t",mesh, 0)
    domains.set_all(1)
    right_domain = AutoSubDomain (lambda x : x[0] >= 0.5 )
    right_domain.mark(domains, 2)
    return domains
def define_mesh_boundaries ( mesh ) :
    boundaries = MeshFunction ("size_t", mesh, 1)

    boundaries.set_all(0)
    class DirichletBCBoundary(SubDomain):
        def inside(self,x, on_boundary):
            return on_boundary
    d_boundary = DirichletBCBoundary()
    d_boundary.mark(boundaries, 1)
    return boundaries

def define_mesh2_boundaries ( mesh ) :
    boundaries= MeshFunction("size_t", mesh, 1)
    boundaries.set_all(0)
    class DirichletBCBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] - 1 < DOLFIN_EPS and x[0] - 0.5 > DOLFIN_EPS
    d_boundary = DirichletBCBoundary()
    d_boundary.mark(boundaries,1)
    class Gamma(SubDomain) :
        def inside(self, x, on_boundary):
            return x[0] > 0.5 - DOLFIN_EPS and x[0] < 0.5 + DOLFIN_EPS
    gamma = Gamma( )
    gamma.mark(boundaries, 2)
    return boundaries

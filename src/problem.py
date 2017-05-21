from __future__ import division
from dolfin import *
import numpy as np
from helpers import *
import sys

n = 10
tau = 0.5
T = 3
# Function definitions
u1_def = "sin(pi*x[0])*sin(pi*x[1])"
phi_def = u1_def
f2_def = "10*(t+1)*sin(pi*x[0]*x[1])"
psi_def = "sin(pi*x[0])*sin(pi*x[1])"
f1_def = "20*t*sin(pi*x[0]*x[1])"
# Defining [0,1]x[0,1] mesh with finite elements of Lagrange type
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "Lagrange", 1)

domains = define_domains(mesh, V, n)

# Create submesh and boundaries for resolving equation on Omega_2
mesh2 = SubMesh(mesh, domains, 2)
boundaries = define_mesh2_boundaries(mesh2)


V2 = FunctionSpace(mesh2, "Lagrange", 1)
u_omega1_layer0 = interpolate(Expression(u1_def), V2)
bcs = [DirichletBC(V2, u_omega1_layer0, boundaries, 2), DirichletBC(V2, 0, boundaries, 1)]

# --- Define measures
ds = Measure("ds")[boundaries]

# -------- define space, finite-element basis function v, and u

u_omega2_layer0 = TrialFunction(V2)
v = TestFunction(V2)
n = FacetNormal(mesh2)
f1 = Expression(f1_def, t=0)
f2 = Expression(f2_def, t=0)
psi = Expression(psi_def)
u_0 = Expression(u1_def)
# ------ Second layer

u2_1 = TrialFunction(V2)
a = u2_1*v*dx + tau*inner(nabla_grad(u2_1), nabla_grad(v))*dx
# a = inner(nabla_grad(u2_1), nabla_grad(v))*dx
u1_1 = interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])", t=tau), V2)
bcs = [DirichletBC(V2, u1_1, boundaries, 2), DirichletBC(V2, 0, boundaries, 1)]
L =  f2*v*dx + inner(grad(u1_1), n)*v*ds(2)

u2_1 = Function(V2)

solve(a == L, u2_1, bcs=bcs)
u1_1 = interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])", t=tau), V)
u_1 = get_whole_function(V, mesh, u1_1, u2_1, domains)

# plot(u_1)

# ------- Layers 2..T



dx = Measure("dx")[domains]
t = 1
boundaries = define_mesh_boundaries(mesh)

u_n = u_1
u_nm1 = u_0
# u_np1 = TrialFunction(V)
# v = TestFunction(V)

bc = DirichletBC(V, 0, boundaries, 1)
f1 = Expression(f1_def, t=t)

while t <= T:
    u_np1 = TrialFunction(V)
    v = TestFunction(V)
    f1.t = t
    f2.t = t
    a = u_np1*v*dx(1) + tau*u_np1*v*dx(2) + (tau**2)*inner(grad(u_np1), grad(v))*dx
    # a = 1/t**2*u_np1*v*dx(1) + inner(grad(u_np1), grad(v))*dx
    L = (2*u_n - u_nm1)*v*dx(1) + tau * u_n * v * dx(2) + tau**2 * f1 * v * dx(1) + tau**2 * f2 * v * dx(2)
    # L = f1*v*dx(1) + f2*v*dx(2) + 1/t**2*(2*u_1 - u_0)*v*dx(1)

    u_np1 = Function(V)
    solve(a == L, u_np1, bcs=bc)
    t += tau
    u_nm1 = u_n
    u_n = u_np1
    p = plot (u_np1, title="t={}".format(t), interactive=False)
    p.write_png('t={}'.format(t))
print norm(u_np1, 'l2')
plot(u_np1, title="t={}".format(t))
print "t = {}".format(t)

interactive()

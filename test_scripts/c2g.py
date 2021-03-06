import numpy as np
import matplotlib.pyplot as plt
import sys

from pydrake.all import MathematicalProgram, Solve, Variables
from pydrake.symbolic import Polynomial

m_p = 1.0
m_c = 10.0
g = 9.8
l = 0.5

prog = MathematicalProgram()
s = prog.NewIndeterminates(1, "s")[0]
c = prog.NewIndeterminates(1, "c")[0]
thetadot = prog.NewIndeterminates(1, "thetadot")[0]
x_cart = prog.NewIndeterminates(1, "x_cart")[0]
xdot_cart = prog.NewIndeterminates(1, "xdot_cart")[0]
# holds the scaling term in front of dynamics functions
# https://piazza.com/class/kk2zncap2s1206?cid=290_f1
z = prog.NewIndeterminates(1, "z")[0]

# new coordinate system: [x, xdot, s, c, thetadot, z]
x = np.array([x_cart, xdot_cart, s, c, thetadot, z])
u = prog.NewIndeterminates(1, "u")[0]

xddot_scaling = z
xddot_term = u + m_p*s*(l*(thetadot**2) + g*c)
thetaddot_scaling = (1/l)*z
thetaddot_term = -u*c - m_p*l*(thetadot**2)*c*s - (m_c + m_p)*g*s
f = [xdot_cart, xddot_scaling*xddot_term,
        c*thetadot, -s*thetadot, thetaddot_scaling*thetaddot_term, -z**2*2*m_p*s*c*thetadot]

# fixed point (resting at the bottom)
x0 = np.array([0, 0, 0, 1, 0, 1/m_c])
# upright
x_star = np.array([0, 0, 0, -1, 0, 1/m_c])

# construct all polynomials containing terms from x up till degree two
deg_V = 6
V_poly = prog.NewFreePolynomial(Variables(x), deg_V)
V = V_poly.ToExpression()

l = x_cart**2 # very simple (temporary) cost function l(x, u)

# Construct the polynomial which is the time derivative of V
Vdot = V.Jacobian(x).dot(f)

# Construct a polynomial L representing the "Lagrange multiplier".
deg_L = 6
L = prog.NewFreePolynomial(Variables(x), deg_L).ToExpression()

# Construct another polynomial L_2 representing the Lagrange multiplier for z
L_2 = prog.NewFreePolynomial(Variables(x), deg_L).ToExpression()

# Add a constraint that Vdot is strictly negative away from x0 (but make an
# exception for the upright fixed point by multipling by s^2).
eps = 1e-18
constraint1 = prog.AddSosConstraint(l + Vdot + L * (s**2 + c**2 - 1) + L_2 * (z*(m_c + m_p*(s**2)) - 1) - eps * (x - x0).dot(x - x0) * s**2)

# Add V(0) = 0 constraint
constraint2 = prog.AddLinearConstraint(
    V.Substitute({
        x_cart: 0,
        xdot_cart: 0,
        s: 0,
        c: 1,
        thetadot: 0,
        z: 1/m_c
    }) == 0)

# integration of V(x) along different axes i think??
int1 = V_poly.Integrate(x_cart, -5, 5)
int2 = int1.Integrate(xdot_cart, -1, 1)
int3 = int2.Integrate(s, -1, 1)
int4 = int3.Integrate(c, -1, 1)
int5 = int4.Integrate(thetadot, -1, 1)
int6 = int5.Integrate(z, 0.1, 1)

# maximize the integral of V(x)
#(TODO: comment this back in if the problem is feasible without this cost)
#prog.AddCost(-1*int6.ToExpression())
print("sahit: this is solving with deg_L=6, deg_V=6, eps=1e-18, both constraints, and no cost") #should try, l,v=4 remove each of the constraints
# Call the solver.
result = Solve(prog)
#assert result.is_success()

print("V =")
Vsol = Polynomial(result.GetSolution(V))
print(Vsol.RemoveTermsWithSmallCoefficients(1e-6))

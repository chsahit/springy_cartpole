import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, Solve, Variables
from pydrake.symbolic import Polynomial

m_p = 1.0
m_c = 10.0
g = 9.8
l = 0.5

prog = MathematicalProgram()
# Declare the "indeterminates", x.  These are the variables which define the
# polynomials, but are NOT decision variables in the optimization.  We will
# add constraints below that must hold FOR ALL x.
s = prog.NewIndeterminates(1, "s")[0]
c = prog.NewIndeterminates(1, "c")[0]
thetadot = prog.NewIndeterminates(1, "thetadot")[0]
x_cart = prog.NewIndeterminates(1, "x_cart")[0]
xdot_cart = prog.NewIndeterminates(1, "xdot_cart")[0]
# holds the scaling term in front of dynamics functions
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
deg_V = 4
V = prog.NewFreePolynomial(Variables(x), deg_V).ToExpression()

l = x_cart**2

# Construct the polynomial which is the time derivative of V
Vdot = V.Jacobian(x).dot(f)

# Construct a polynomial L representing the "Lagrange multiplier".
deg_L = 4
L = prog.NewFreePolynomial(Variables(x), deg_L).ToExpression()

# Construct another polynomial L_2 representing the Lagrange multiplier for z
L_2 = prog.NewFreePolynomial(Variables(x), deg_L).ToExpression()

# Add a constraint that Vdot is strictly negative away from x0 (but make an
# exception for the upright fixed point by multipling by s^2).
eps = 1e-4
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

# Call the solver.
result = Solve(prog)
#assert result.is_success()

print("V =")
Vsol = Polynomial(result.GetSolution(V))
print(Vsol.RemoveTermsWithSmallCoefficients(1e-6))

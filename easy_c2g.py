import numpy as np
import matplotlib.pyplot as plt
import sys

from pydrake.all import MathematicalProgram, Solve, Variables
from pydrake.symbolic import Polynomial

import utils

def compute_lyapunov_function(deg_V = 4, deg_L = 4, mode="no_contact"):
    m_p, m_c, g, l, k, x_01, x_02 = utils.get_system_parameters()

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

    f = utils.compute_dynamics(*x, u, mode=mode)

    # fixed point (resting at the bottom)
    x0 = np.array([0, 0, 0, 1, 0, 1/m_c])
    # upright
    x_star = np.array([0, 0, 0, -1, 0, 1/m_c])

    # construct all polynomials containing terms from x up till degree two
    V_poly = prog.NewFreePolynomial(Variables(x), deg_V)
    V = V_poly.ToExpression()

    loss = (x_cart**2) + 10*(c+1)**2 + 0.5*(thetadot**2)

    # Construct the polynomial which is the time derivative of V
    Vdot = V.Jacobian(x).dot(f)

    # Construct a polynomial L representing the "Lagrange multiplier".
    L = prog.NewFreePolynomial(Variables(x), deg_L).ToExpression()

    # Construct another polynomial L_2 representing the Lagrange multiplier for z
    L_2 = prog.NewFreePolynomial(Variables(x), deg_L).ToExpression()

    # Add a constraint that Vdot is strictly negative away from x0 (but make an
    # exception for the upright fixed point by multipling by s^2).
    eps = 1e-4
    constraint1 = prog.AddSosConstraint(loss + Vdot + L * (s**2 + c**2 - 1) + L_2 * (z*(m_c + m_p*(s**2)) - 1) - eps * (x - x0).dot(x - x0) * s**2)

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
    int2 = int1.Integrate(xdot_cart, -2, 2)
    int3 = int2.Integrate(s, -1, 1)
    int4 = int3.Integrate(c, -1, 1)
    int5 = int4.Integrate(thetadot, -2, 2)
    int6 = int5.Integrate(z, 0.1, 1)

    # maximize the integral of V(x)
    prog.AddCost(-1*int6.ToExpression())
    print(f"Solving with deg_L={deg_L}, deg_V={deg_V}, eps={eps}")
    # Call the solver.
    decision_vars = prog.decision_variables()
    prog.AddBoundingBoxConstraint(-50, 50, np.array([decision_vars]))
    result = Solve(prog)

    assert result.is_success()

    #print("V =")
    Vsol = Polynomial(result.GetSolution(V))
    """
    int1 = Vsol.Integrate(x_cart, -15, 15)
    int2 = int1.Integrate(xdot_cart, -10, 10)
    int3 = int2.Integrate(s, -1, 1)
    int4 = int3.Integrate(c, -1, 1)
    int5 = int4.Integrate(thetadot, -20, 20)
    int6 = int5.Integrate(z, 0.1, 1)
    """
    integral = utils.integrate_c2g(Vsol)
    #print(Vsol.RemoveTermsWithSmallCoefficients(1e-1))
    print("integral: ", integral)

    return Vsol

if __name__ == "__main__":
    compute_lyapunov_function()

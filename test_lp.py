# given a state x cost-to-go V, and dynamics f solve for a minimizing u
import numpy as np
import matplotlib.pyplot as plt
import sys

from pydrake.all import MathematicalProgram, Solve, Variables, Variable
from pydrake.symbolic import Polynomial

from easy_c2g import compute_lyapunov_function

# dynamics parameters
m_c = 10.0
m_p = 1.0
l = 0.5
g = 9.8

# copied from easy_c2g, i should make this a util function
# somewhere
def compute_dynamics(x_cart, xdot_cart, s, c, thetadot, z, u):
    xddot_term = u + m_p*s*(l*(thetadot**2) + g*c)
    thetaddot_scaling = (1/l)*z
    thetaddot_term = -u*c - m_p*l*(thetadot**2)*c*s - (m_c + m_p)*g*s
    f = [xdot_cart, z*xddot_term, c*thetadot, -s*thetadot,
            thetaddot_scaling*thetaddot_term, -z**2*2*m_p*s*c*thetadot]
    return f


#state
state_dict = {"x_cart(0)": 0, "xdot_cart(0)" : 0, "s(0)": np.sqrt(2)/2, "c(0)": np.sqrt(2)/2, "thetadot(0)": 0, "z(0)" : 1/m_c}
x_cart = 0
xdot_cart = 0
s = np.sqrt(2)/2
c = np.sqrt(2)/2
thetadot = 0
z = 1/m_c

V_poly = compute_lyapunov_function()
V = V_poly.ToExpression()

prog = MathematicalProgram()
variables = list()
vars_dict = dict()
for v in V.GetVariables():
    variables.append(v)
    vars_dict[v] = state_dict[v.get_name()]
x = np.array(variables)
u = prog.NewContinuousVariables(1, "u")[0]

f = compute_dynamics(x_cart, xdot_cart, s, c, thetadot, z, u)
Vdot = V.Jacobian(x).dot(f)
subbed = Vdot.Substitute(vars_dict)
prog.AddCost(subbed)
prog.AddBoundingBoxConstraint(-10, 10, u)
result = Solve(prog)
print("final control eq: ", subbed)
assert result.is_success()
control = result.GetSolution(u)
print("control: ", control)


import numpy as np
import matplotlib.pyplot as plt
from easy_c2g import compute_lyapunov_function
from pydrake.all import MathematicalProgram, Solve, Variables, Variable
from pydrake.symbolic import Polynomial
m_c = 10
m_p = 1
g = 9.8
l = np.pi - 0.85
variable_order = {"x_cart(0)": 0, "xdot_cart(0)": 1, "s(0)": 2, "c(0)": 3, "thetadot(0)": 4, "z(0)": 5}

class MockLyapunovIO():
    def __init__(self, cost_to_go):
        self.cost_to_go = cost_to_go

    # map from [x, theta, xdot, thetadot] to [x, xdot, s, c, thetadot, z]
    def cost_to_go_state(self, cartpole_state):
        theta = cartpole_state[1]
        c = np.cos(theta)
        s = np.sin(theta)
        z = 1/(m_c + m_p*s**2)
        # xm xdot
        c2g_state = np.array([cartpole_state[0], cartpole_state[2], s, c, cartpole_state[3], z])
        return c2g_state

    def c2g_dict_variables(self, c2g_state, c2g):
        state_dict = {"x_cart(0)": c2g_state[0],
                "xdot_cart(0)": c2g_state[1], "s(0)": c2g_state[2], "c(0)": c2g_state[3],
                "thetadot(0)": c2g_state[4], "z(0)": c2g_state[5]}
        c2g_var_dict = dict()
        variables = [None]*6
        for v in c2g.GetVariables():
            variables[variable_order[v.get_name()]] = v
            c2g_var_dict[v] = state_dict[v.get_name()]
        print(variables)
        return variables, c2g_var_dict

    def compute_dynamics(self, x_cart, xdot_cart, s, c, thetadot, z, u):
        xddot_term = u + m_p*s*(l*(thetadot**2) + g*c)
        thetaddot_scaling = (1/l)*z
        thetaddot_term = -u*c - m_p*l*(thetadot**2)*c*s - (m_c + m_p)*g*s
        f = [xdot_cart, z*xddot_term, c*thetadot, -s*thetadot,
                thetaddot_scaling*thetaddot_term, -z**2*2*m_p*s*c*thetadot]
        return f

    def get_lyapunov_output(self, cartpole_state):
        c2g_state = self.cost_to_go_state(cartpole_state)
        variables, c2g_var_dict = self.c2g_dict_variables(c2g_state, self.cost_to_go)
        prog = MathematicalProgram()
        u = prog.NewContinuousVariables(1, "u")[0]
        f = self.compute_dynamics(*c2g_state, u)
        Vjac = self.cost_to_go.Jacobian(variables)
        #print("Vjac: ", Vjac[1])
        Vdot = self.cost_to_go.Jacobian(variables).dot(f)
        #print(Vdot)
        Vdot_at_state = Vdot.Substitute(c2g_var_dict)
        c2g_var_dict_copy = c2g_var_dict.copy()
        c2g_var_dict_copy.pop(variables[3])
        Vdot_at_partial_state = Vdot.Substitute(c2g_var_dict_copy)
        print("partial vdot: ", Vdot_at_partial_state.Expand())
        V_at_state = self.cost_to_go.Substitute(c2g_var_dict)
        #print("c2g = ", V_at_state)
        prog.AddCost(Vdot_at_state)
        prog.AddBoundingBoxConstraint(-15, 15, u)
        result = Solve(prog)
        #print("vdot if 15: ", Vdot_at_state.Substitute({u: 15}))
        #print("vdot if -15: ", Vdot_at_state.Substitute({u: -15}))
        assert result.is_success()
        control = result.GetSolution(u)
        #print("picking: ", control)
        return float(V_at_state.to_string()), control

    def compute_V_outputs(self):
        x = np.linspace(-10, 10)
        V_outputs = list()
        control_outputs = list()
        for i in range(x.shape[0]):
            state = np.array([x[i], np.pi, 0, 0])
            outputs = self.get_lyapunov_output(state)
            V_outputs.append(outputs[0])
            control_outputs.append(outputs[1])
        return x, np.array(V_outputs), np.array(control_outputs)


if __name__ == "__main__":

    V_poly = compute_lyapunov_function(deg_V=2, deg_L=2)
    #V_poly = V_poly.RemoveTermsWithSmallCoefficients(1e-6)
    #print(V_print)
    V = V_poly.ToExpression()
    lyapunov_mocker = MockLyapunovIO(V)
    states,V_outputs, control_outputs = lyapunov_mocker.compute_V_outputs()
    plt.plot(states, control_outputs)
    plt.show()
    plt.plot(states, V_outputs)
    plt.show()

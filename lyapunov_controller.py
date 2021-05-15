import numpy as np
from pydrake.systems.framework import LeafSystem, BasicVector, PortDataType
from pydrake.common.value import Value
from pydrake.systems.primitives import FirstOrderTaylorApproximation
from pydrake.all import MathematicalProgram, Solve, Variables, Variable
from pydrake.symbolic import Polynomial
m_c = 10
m_p = 1
g = 9.8
l = np.pi - 0.85
variable_order = {"x_cart(0)": 0, "xdot_cart(0)": 1, "s(0)": 2, "c(0)": 3, "thetadot(0)": 4, "z(0)": 5}

class LyapunovCartpoleController(LeafSystem):
    def __init__(self, plant, cost_to_go):
        LeafSystem.__init__(self)
        self.use_lqr = False
        self.cost_to_go = cost_to_go
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._positions_port = self.DeclareVectorInputPort("configuration", BasicVector(8))
        self.DeclareVectorOutputPort("controls", BasicVector(1), self.CalcOutput)
        self._K = np.array([[ -3.16227766, 254.39247154,  -9.76923594,  55.17739406]])

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
        variables = [None] * 6
        for v in c2g.GetVariables():
            variables[variable_order[v.get_name()]] = v
            c2g_var_dict[v] = state_dict[v.get_name()]
        return variables, c2g_var_dict

    def compute_dynamics(self, x_cart, xdot_cart, s, c, thetadot, z, u):
        xddot_term = u + m_p*s*(l*(thetadot**2) + g*c)
        thetaddot_scaling = (1/l)*z
        thetaddot_term = -u*c - m_p*l*(thetadot**2)*c*s - (m_c + m_p)*g*s
        f = [xdot_cart, z*xddot_term, c*thetadot, -s*thetadot,
                thetaddot_scaling*thetaddot_term, -z**2*2*m_p*s*c*thetadot]
        return f

    def get_lyapunov_control(self, cartpole_state):
        c2g_state = self.cost_to_go_state(cartpole_state)
        variables, c2g_var_dict = self.c2g_dict_variables(c2g_state, self.cost_to_go)
        prog = MathematicalProgram()
        u = prog.NewContinuousVariables(1, "u")[0]
        f = self.compute_dynamics(*c2g_state, u)
        Vdot = self.cost_to_go.Jacobian(variables).dot(f)
        Vdot_at_state = Vdot.Substitute(c2g_var_dict)
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
        return control

    def CalcOutput(self, context, output):
        state = self._positions_port.Eval(context)
        cartpole_state = np.array([state[0], state[1], state[4], state[5]])
        #lqr output
        lqr_state = np.copy(cartpole_state)
        lqr_state[1] = lqr_state[1] - np.pi
        tau_lqr = -np.matmul(self._K, lqr_state)
        if np.abs(np.cos(state[1]) + 1) < 0.00 and np.abs(state[0]) < 0.3:
            self.use_lqr = True
            tau = tau_lqr
        else:
            tau = np.array([self.get_lyapunov_control(cartpole_state)])
        tau = np.clip(tau, -15.0, 15.0)
        output.SetFromVector(tau)

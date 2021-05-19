import numpy as np
from pydrake.systems.framework import LeafSystem, BasicVector, PortDataType
from pydrake.common.value import Value
from pydrake.systems.primitives import FirstOrderTaylorApproximation
from pydrake.all import MathematicalProgram, Solve, Variables, Variable
from pydrake.symbolic import Polynomial
import utils

m_c, m_p, g, l, k, x_01, x_02 = utils.get_system_parameters()
variable_order = utils.variable_order
modes = ["no_contact", "lc", "lp", "rc", "rp"]

class LyapunovCartpoleController(LeafSystem):
    def __init__(self, plant, cost_to_gos):
        LeafSystem.__init__(self)
        self.use_lqr = False
        self.cost_to_gos = cost_to_gos
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

    """
    given a cost_to_go function and an appropriately formatted state
    get a list form of the variables to compute the jacobian
    and get a dictionary form mapping variables to state values
    to perform substitutions
    """
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

    def get_lyapunov_control(self, cartpole_state):
        mode = self.compute_mode(cartpole_state[0], cartpole_state[1])
        c2g_state = self.cost_to_go_state(cartpole_state)
        variables, c2g_var_dict = self.c2g_dict_variables(c2g_state, self.cost_to_gos[mode])
        #prog = MathematicalProgram()
        #u = prog.NewContinuousVariables(1, "u")[0]
        u = Variable("u")
        f = utils.compute_dynamics(*c2g_state, u)
        Vdot = self.cost_to_gos[mode].Jacobian(variables).dot(f)
        Vdot_at_state = Vdot.Substitute(c2g_var_dict)
        #V_at_state = self.cost_to_gos[mode].Substitute(c2g_var_dict)
        output_1 = Vdot_at_state.Substitute({u: 30})
        output_2 = Vdot_at_state.Substitute({u: -30})
        if float(output_1.to_string()) < float(output_2.to_string()):
            return 30
        else:
            return -30
        #return control
        #print("c2g = ", V_at_state)
        #prog.AddCost(Vdot_at_state)
        #prog.AddBoundingBoxConstraint(-15, 15, u)
        #result = Solve(prog)
        #print("vdot if 15: ", Vdot_at_state.Substitute({u: 15}))
        #print("vdot if -15: ", Vdot_at_state.Substitute({u: -15}))
        #assert result.is_success()
        #control = result.GetSolution(u)
        #print("picking: ", control)

    def compute_mode(self, x, theta):
        if x < x_01:
            return 1
        elif x + l*np.sin(theta) < x_01:
            return 2
        elif x > x_02:
            return 3
        elif x + l*np.sin(theta) > x_02:
            return 4
        else:
            return 0

    def CalcOutput(self, context, output):
        state = self._positions_port.Eval(context)
        cartpole_state = np.array([state[0], state[1], state[4], state[5]])

        #lqr output
        lqr_state = np.copy(cartpole_state)
        lqr_state[1] = lqr_state[1] - np.pi
        tau_lqr = -np.matmul(self._K, lqr_state)
        if np.abs(np.cos(state[1]) + 1) < 0.5 and np.abs(state[0]) < 0.5:
            self.use_lqr = True
            tau = tau_lqr
        else:
            tau = np.array([self.get_lyapunov_control(cartpole_state)])
        tau = np.clip(tau, -25.0, 25.0)
        output.SetFromVector(tau)

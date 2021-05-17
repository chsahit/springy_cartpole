import numpy as np
from pydrake.systems.framework import LeafSystem, BasicVector, PortDataType
from pydrake.common.value import AbstractValue, Value
from pydrake.systems.primitives import FirstOrderTaylorApproximation
m_cart = 10
m_ball = 1
g = 9.8
l = np.pi - 0.85
#l = 0.5
desired_energy = m_ball*g*l
class CartpoleController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.use_lqr = False
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._positions_port = self.DeclareVectorInputPort("configuration", BasicVector(8))
        self.DeclareVectorOutputPort("controls", BasicVector(1), self.CalcOutput)
        self._K = np.array([[ -3.16227766, 254.39247154,  -9.76923594,  55.17739406]])
        self.theta_dots = list()
        self.thetas = list()
        self.energies = list()

    def E(self, theta, theta_dot):
        return 0.5*m_ball*(theta_dot**2) - m_ball * g * l * np.cos(theta)

    def f(self, xddot_des, theta, theta_dot):
        c = np.cos(theta)
        s = np.sin(theta)
        f_x = (2 - c**2)*xddot_des - s*c - (theta_dot**2)*s
        return f_x

    def CalcOutput(self, context, output):
        state = self._positions_port.Eval(context)
        cartpole_state = np.array([state[0], state[1], state[4], state[5]])
        cartpole_state[1] = cartpole_state[1] - np.pi
        theta = state[1]
        theta_dot = state[5]
        #lqr output
        tau_lqr = -np.matmul(self._K, cartpole_state)
        # energy shaping output:
        tau_energy = 1.0 * np.array([theta_dot*np.cos(theta)*(self.E(theta, theta_dot) - desired_energy) - 200*state[0] - 200*state[4]])
        if np.abs(np.cos(theta) + 1) < 0.5 and np.abs(state[0]) < 0.5:
            self.use_lqr = True
            tau = tau_lqr
            #print("lqr")
        else:
            tau = self.f(tau_energy, theta, theta_dot)
        print("energy ", theta)
        tau = np.clip(tau, -15.0, 15.0)
        self.theta_dots.append(0.5*(theta_dot**2))
        self.thetas.append(-np.pi*g*np.cos(theta))
        self.energies.append(self.E(theta, theta_dot))
        output.SetFromVector(tau)

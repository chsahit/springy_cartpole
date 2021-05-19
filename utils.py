from pydrake.symbolic import Polynomial
def get_system_parameters():
    #m_p, m_c, g, l, K (wall stiffness),
    # resting point (left spring), resting point right spring
    return 1.0, 10.0, 9.8, 0.5, 5, -1.5+0.12, 1.5-0.12,

def compute_dynamics(x_cart, xdot_cart, s, c, thetadot, z, u, mode="no_contact"):
    m_p, m_c, g, l, k, x_01, x_02 = get_system_parameters()
    xddot_scaling = z
    xddot_term = u + m_p*s*(l*(thetadot**2) + g*c)
    thetaddot_scaling = (1/l)*z
    thetaddot_term = -u*c - m_p*l*(thetadot**2)*c*s - (m_c + m_p)*g*s
    additional_thetaddot_term = 0
    if mode == "cart_left":
        xddot_term += k * (x_01 - x_cart)
        thetaddot_term += -c * k * (x_01 - x_cart)
    elif mode == "cart_right":
        xddot_term += k * (x_02 - x_cart)
        thetaddot_term += -c * k * (x_02 - x_cart)
    elif mode == "pole_left":
        xddot_term = -k*(c**2)*(x_01 - x_cart - l*s)
        thetaddot_term += -c*k*(x_01 - x_cart - l*s)
        additional_thetaddot_numerator = (m_c + m_p)*k*c*(x_01 - x_cart - l*s)*z
        additional_thetaddot_denom = m_p*l
        additional_thetaddot_term = additional_thetaddot_numerator/additional_thetaddot_denom
    elif mode == "pole_right":
        xddot_term = -k*(c**2)*(x_02 - x_cart - l*s)
        thetaddot_term += -c*k*(x_02 - x_cart - l*s)
        additional_thetaddot_numerator = (m_c + m_p)*k*c*(x_02 - x_cart - l*s)*z
        additional_thetaddot_denom = m_p*l
        additional_thetaddot_term = additional_thetaddot_numerator/additional_thetaddot_denom


    f = [xdot_cart, xddot_scaling*xddot_term,
            c*thetadot, -s*thetadot,
            thetaddot_scaling*thetaddot_term + additional_thetaddot_term,
            -z**2*2*m_p*s*c*thetadot]
    return f

# maybe split these up, ie have one utils for dynamics and one for
# working with functions

variable_order = {"x_cart(0)": 0, "xdot_cart(0)": 1, "s(0)": 2, "c(0)": 3, "thetadot(0)": 4, "z(0)": 5}

def get_variable_list(c2g):
    variables = [None] * 6
    for v in c2g.GetVariables():
        variables[variable_order[v.get_name()]] = v
    return variables

def integrate_c2g(V, x_cart_min=-15, x_cart_max=15, variables=None):
    if variables is None:
        variables = get_variable_list(V.ToExpression())
    int1 = V.Integrate(variables[0], x_cart_min, x_cart_max)
    int2 = int1.Integrate(variables[1], -10, 10)
    int3 = int2.Integrate(variables[2], -1, 1)
    int4 = int3.Integrate(variables[3], -1, 1)
    int5 = int4.Integrate(variables[4], -20, 20)
    int6 = int5.Integrate(variables[5], 0.1, 1)
    return int6


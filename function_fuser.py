import numpy as np
import utils
from pydrake.all import MathematicalProgram, Solve, Variables
from pydrake.symbolic import Polynomial
def fuse_functions(V_no_contact, V_left_cart, deg_V=2):
    prog = MathematicalProgram()
    s = prog.NewIndeterminates(1, "s")[0]
    c = prog.NewIndeterminates(1, "c")[0]
    thetadot = prog.NewIndeterminates(1, "thetadot")[0]
    x_cart = prog.NewIndeterminates(1, "x_cart")[0]
    xdot_cart = prog.NewIndeterminates(1, "xdot_cart")[0]
    z = prog.NewIndeterminates(1, "z")[0]
    x = np.array([x_cart, xdot_cart, s, c, thetadot, z])
    V_no_contact_new = prog.NewFreePolynomial(Variables(x), deg_V)
    V_left_cart_new = prog.NewFreePolynomial(Variables(x), deg_V)

    left_cart_new_partial_int = utils.integrate_c2g(V_left_cart_new, x_cart_min=-1.6,
            x_cart_max=-1.4, variables=x)
    no_contact_new_partial_int = utils.integrate_c2g(V_no_contact_new, x_cart_min=-1.6, \
            x_cart_max=-1.4, variables=x)
    change_left_cart_no_contact = left_cart_new_partial_int - no_contact_new_partial_int

    no_contact_new_monom_map = V_no_contact_new.monomial_to_coefficient_map()
    no_contact_monom_map = V_no_contact.monomial_to_coefficient_map()
    left_cart_new_monom_map = V_left_cart_new.monomial_to_coefficient_map()
    left_cart_monom_map = V_left_cart.monomial_to_coefficient_map()

    prog.AddCost((change_left_cart_no_contact**2).ToExpression())
    prog.AddBoundingBoxConstraint(-50, 50, np.array([prog.decision_variables()]))
    # TODO: the new maps are emptier than the old ones!!
    # honestly if i cant use these as a seed maybe i can use them as is
    for monomial in left_cart_new_monom_map.keys():
        new_coeff = left_cart_new_monom_map[monomial]
        old_coeff = left_cart_monom_map.get(monomial, 0)
        prog.AddConstraint((new_coeff - old_coeff)**2 <= 1e-10)
    for monomial in no_contact_new_monom_map.keys():
        new_coeff = no_contact_new_monom_map[monomial]
        old_coeff = no_contact_monom_map.get(monomial, 0)
        prog.AddConstraint((new_coeff - old_coeff)**2 <= 1e-10)


    print("solving fusion(ish)")
    result = Solve(prog)

    V_left_cart_new = Polynomial(result.GetSolution(V_left_cart_new.ToExpression()))
    V_no_contact_new = Polynomial(result.GetSolution(V_no_contact_new.ToExpression()))
    left_partial_int = utils.integrate_c2g(V_left_cart_new, x_cart_min=-1.6, x_cart_max=-1.4)
    free_partial_int = utils.integrate_c2g(V_no_contact_new, x_cart_min=-1.6, x_cart_max=-1.4)
    print("difference in news: ", left_partial_int - free_partial_int)

    left_full_int = utils.integrate_c2g(V_left_cart_new)
    old_left_full_int = utils.integrate_c2g(V_left_cart)
    print("difference in left: ", left_full_int - old_left_full_int)

    free_full_int = utils.integrate_c2g(V_no_contact_new)
    old_free_full_int = utils.integrate_c2g(V_no_contact)
    print("difference in free: ", free_full_int - old_free_full_int)

    import IPython; IPython.embed()

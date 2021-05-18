import numpy as np
import utils
from pydrake.all import MathematicalProgram, Solve, Variables
from pydrake.symbolic import Polynomial
from pydrake.common.containers import EqualToDict
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
    print("lc new: ", V_left_cart_new)
    no_contact_monom_map = V_no_contact.monomial_to_coefficient_map()
    left_cart_new_monom_map = V_left_cart_new.monomial_to_coefficient_map()
    left_cart_monom_map = V_left_cart.monomial_to_coefficient_map()

    prog.AddCost((change_left_cart_no_contact**2).ToExpression())
    prog.AddBoundingBoxConstraint(-50, 50, np.array([prog.decision_variables()]))
    # TODO: the new maps are emptier than the old ones!!
    # honestly if i cant use these as a seed maybe i can use them as is
    print("lc new keys: ", list(no_contact_new_monom_map.keys()))
    print("lc old keys: ", list(left_cart_monom_map.keys()))

    for monomial in left_cart_new_monom_map.keys():
        new_coeff = left_cart_new_monom_map[monomial]
        old_coeff = left_cart_monom_map.get(monomial, 0)
        prog.AddConstraint((new_coeff - old_coeff)**2 <= 1e-10)
    for monomial in no_contact_new_monom_map.keys():
        new_coeff = no_contact_new_monom_map[monomial]
        old_coeff = no_contact_monom_map.get(monomial, 0)
        prog.AddConstraint((new_coeff - old_coeff)**2 <= 1e-10)

    add_coefficient_constraints(prog, V_no_contact, V_no_contact_new)
    print("solving fusion(ish)")
    result = Solve(prog)

    V_left_cart_new = Polynomial(result.GetSolution(V_left_cart_new.ToExpression()))
    V_no_contact_new = Polynomial(result.GetSolution(V_no_contact_new.ToExpression()))
    print("lc new solve: ", V_left_cart_new)
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

# EqualToDict isn't really cooperating

"""
Monomials don't match just by having the same name, Maybe the more Drake friendly
way to do this would be to mess with variable IDs.... For now this does finds
a monomial in a polynomial by comparing string names
"""
def find_matching_monomial(monomial, polynomial):
    given_monomial_vars = list(monomial.GetVariables())
    given_monomial_varnames = [x.get_name() for x in given_monomial_vars]
    given_monomial_powers = EqualToDict(monomial.get_powers())
    poly_map = polynomial.monomial_to_coefficient_map()
    for curr_monom in poly_map.keys():
        curr_monom_vars = list(monomial.GetVariables())
        curr_monom_varnames = [x.get_name() for x in curr_monom_vars]
        curr_monom_powers = EqualToDict(curr_monom.get_powers())
        if set(curr_monom_varnames) != set(given_monomial_varnames):
            continue
        powers_match = True
        for curr_varname in curr_monom_varnames:
            curr_var = curr_monom_vars[curr_monom_varnames.index(curr_varname)]
            given_var = given_monomial_vars[given_monomial_varnames.index(curr_varname)]
            #print("items: ", list(curr_monom_powers.items()))
            print("curr name: ", curr_varname)
            #print("given power: ", given_monomial_powers[given_var])
            curr_power = 1
            if given_monomial_powers[given_var] != curr_power:
                powers_match = False
                break
        if powers_match:
            return curr_monom

# This and find_matching_monomial are sooooo
# inefficent.
def add_coefficient_constraints(prog, f1, f2):
    f1_coeff_map = f1.monomial_to_coefficient_map()
    f2_coeff_map = f2.monomial_to_coefficient_map()
    for f1_monom in f1_coeff_map.keys():
        f2_monom = find_matching_monomial(f1_monom, f2)
        print("f1: ", f1_monom)
        print("f2: ", f2_monom)


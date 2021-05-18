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
    no_contact_monom_map = V_no_contact.monomial_to_coefficient_map()
    left_cart_new_monom_map = V_left_cart_new.monomial_to_coefficient_map()
    left_cart_monom_map = V_left_cart.monomial_to_coefficient_map()

    prog.AddCost((change_left_cart_no_contact**2).ToExpression())
    prog.AddBoundingBoxConstraint(-50, 50, np.array([prog.decision_variables()]))
    add_coefficient_constraints(prog, V_no_contact, V_no_contact_new)
    add_coefficient_constraints(prog, V_left_cart, V_left_cart_new)
    print("solving fusion(ish)")
    result = Solve(prog)

    V_left_cart_new = Polynomial(result.GetSolution(V_left_cart_new.ToExpression()))
    V_no_contact_new = Polynomial(result.GetSolution(V_no_contact_new.ToExpression()))

    print_diagnostic_info(V_no_contact, V_left_cart, V_no_contact_new, V_left_cart_new)

# EqualToDict isn't really cooperating
def compare_monomial_powers(m1, m2):
    m1_power_map = list(m1.items())
    m2_power_map = list(m2.items())
    m1_power_map = [(k.get_name(), v) for k, v in m1_power_map]
    m2_power_map = [(k.get_name(), v) for k, v in m2_power_map]
    return set(m1_power_map) == set(m2_power_map)

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
        curr_monom_powers = curr_monom.get_powers()
        if set(curr_monom_varnames) != set(given_monomial_varnames):
            continue
        equalmonoms = compare_monomial_powers(curr_monom_powers, given_monomial_powers)
        if equalmonoms:
            return curr_monom

def add_coefficient_constraints(prog, f1, f2):
    f1_coeff_map = f1.monomial_to_coefficient_map()
    f2_coeff_map = f2.monomial_to_coefficient_map()
    for f1_monom in f1_coeff_map.keys():
        f2_monom = find_matching_monomial(f1_monom, f2)
        f1_coeff = f1_coeff_map[f1_monom]
        f2_coeff = f2_coeff_map[f2_monom]
        prog.AddConstraint((f1_coeff - f2_coeff)**2 <= 1e-20)

def print_diagnostic_info(old_f1, old_f2, new_f1, new_f2):
    old_f1_partial_int = utils.integrate_c2g(old_f1, x_cart_min=-1.6, x_cart_max=-1.4)
    old_f2_partial_int = utils.integrate_c2g(old_f2, x_cart_min=-1.6, x_cart_max=-1.4)
    print("difference in olds: ", old_f1_partial_int - old_f2_partial_int)

    new_f1_partial_int = utils.integrate_c2g(new_f1, x_cart_min=-1.6, x_cart_max=-1.4)
    new_f2_partial_int = utils.integrate_c2g(new_f2, x_cart_min=-1.6, x_cart_max=-1.4)
    print("difference in news: ", new_f1_partial_int - new_f2_partial_int)

    old_f1_full_int = utils.integrate_c2g(old_f1)
    new_f1_full_int = utils.integrate_c2g(new_f1)
    print("difference in f1s: ", old_f1_full_int - new_f1_full_int)

    old_f2_full_int = utils.integrate_c2g(old_f2)
    new_f2_full_int = utils.integrate_c2g(new_f2)
    print("difference in f2s: ", old_f2_full_int - new_f2_full_int)



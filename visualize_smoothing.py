from visualize_lyapunov import visualize_two
from easy_c2g import compute_lyapunov_function
import function_fuser

V_free = compute_lyapunov_function(deg_V=2, deg_L=2)
V_cart_left = compute_lyapunov_function(deg_V=2, deg_L=2, mode="cart_left")

V1smooth, V2smooth = function_fuser.fuse_functions(V_free, V_cart_left, deg_V=2)

visualize_two(V_free, V_cart_left, show=False)
visualize_two(V1smooth, V2smooth, show=True)

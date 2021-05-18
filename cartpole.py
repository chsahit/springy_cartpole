import numpy as np
import sys
from controllers.energy_controller import CartpoleController
from lyapunov_controller import LyapunovCartpoleController
from easy_c2g import compute_lyapunov_function
import function_fuser

from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import FirstOrderTaylorApproximation
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import PrismaticJoint, LinearSpringDamper
from pydrake.common import FindResourceOrThrow

import matplotlib.pyplot as plt

stiffness = 12
base_wall_offset = 3.8

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def setup_walls(plant):
    wall_right = Parser(plant).AddModelFromFile("wall.sdf")
    spring_wall_right = Parser(plant).AddModelFromFile("wall.sdf", "springy_wall_right")
    plant.WeldFrames(
        frame_on_parent_P=plant.world_frame(),
        frame_on_child_C=plant.GetFrameByName("Wall", wall_right),
        X_PC=xyz_rpy_deg([base_wall_offset, 0, 0], [0, 0, 0]))
    right_joint = PrismaticJoint(
        "right_joint", plant.GetFrameByName("Wall", wall_right),
        plant.GetFrameByName("Wall", spring_wall_right), np.array([1, 0, 0]))
    right_joint.set_default_translation(-2.3)
    plant.AddJoint(right_joint)
    right_spring = LinearSpringDamper(
        plant.GetBodyByName("Wall", wall_right), np.array([0, 0, 0]),
        plant.GetBodyByName("Wall", spring_wall_right), np.array([0, 0, 0]),
        2.3, stiffness, 0.0)
    plant.AddForceElement(right_spring)

    wall_left = Parser(plant).AddModelFromFile("wall.sdf", "wall_left")
    spring_wall_left = Parser(plant).AddModelFromFile("wall.sdf", "springy_wall_left")
    plant.WeldFrames(
        frame_on_parent_P=plant.world_frame(),
        frame_on_child_C=plant.GetFrameByName("Wall", wall_left),
        X_PC=xyz_rpy_deg([-base_wall_offset, 0, 0], [0, 0, 0]))
    left_joint = PrismaticJoint(
        "left_joint", plant.GetFrameByName("Wall", wall_left),
        plant.GetFrameByName("Wall", spring_wall_left), np.array([1, 0, 0]))
    left_joint.set_default_translation(2.3)
    plant.AddJoint(left_joint)
    left_spring = LinearSpringDamper(
        plant.GetBodyByName("Wall", wall_left), np.array([0, 0, 0]),
        plant.GetBodyByName("Wall", spring_wall_left), np.array([0, 0, 0]),
        2.3, stiffness, 0.0)
    plant.AddForceElement(left_spring)


V_free = compute_lyapunov_function(deg_V=2, deg_L=2)
V_cart_left = compute_lyapunov_function(deg_V=2, deg_L=2, mode="cart_left")
V_cart_right = compute_lyapunov_function(deg_V=2, deg_L=2, mode="cart_right")
V_pole_left = compute_lyapunov_function(deg_V=2, deg_L=2, mode="pole_left")
V_pole_right = compute_lyapunov_function(deg_V=2, deg_L=2, mode="pole_right")

function_fuser.fuse_functions(V_free, V_cart_left)
sys.exit()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
cartpole = Parser(plant).AddModelFromFile("cart_pole.sdf")
setup_walls(plant)
plant.Finalize()

# Setup visualization
visualizer = ConnectMeshcatVisualizer(
    builder,
    scene_graph=scene_graph,
    zmq_url="new")
visualizer.vis.delete()
visualizer.set_planar_viewpoint(xmin=-2.5, xmax=2.5, ymin=-1.0, ymax=2.5)
#controller = builder.AddSystem(CartpoleController(plant))
controller = builder.AddSystem(LyapunovCartpoleController(plant, V_free.ToExpression()))
builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))
builder.Connect(controller.get_output_port(0),plant.get_actuation_input_port())
diagram = builder.Build()
# Set up a simulator to run this diagram
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyContextFromRoot(context)
#plant.get_actuation_input_port(cartpole).FixValue(plant_context, np.array([0]))
# Set the initial conditions
context.SetContinuousState([1.0, np.pi - 0.5, -2.3, 2.3, 0.0, 0.0, 0, 0.00]) # x, theta, wall1, wall2, xdot, thetadot, wall1dot, wall2dot
context.SetTime(0.0)

visualizer.start_recording()
simulator.AdvanceTo(26.0)
visualizer.publish_recording()
visualizer.vis.render_static()
print("on input")
input()

"""
plt.plot(controller.theta_dots)
plt.plot(controller.thetas)
plt.plot(controller.energies)
"""
plt.show()

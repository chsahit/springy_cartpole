import matplotlib.pyplot as plt
import pickle
import numpy as np

opt_x, opt_theta = pickle.load(open("notfused.p", "rb"))
energy_x, energy_theta = pickle.load(open("fused.p", "rb"))

plt.title("Trajectory of Cartpole")
plt.plot([0], [np.pi], 'o', color='g')
plt.plot([-1.0], [np.pi - 0.5], 'o', color='r')
plt.xlabel("x position")
plt.ylabel("theta")
plt.plot(opt_x, opt_theta)
plt.plot(energy_x, energy_theta)
plt.show()


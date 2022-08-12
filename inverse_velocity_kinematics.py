#  Copyright (c) Miguel L. Rodrigues 2022.

import numpy as np

from lib.forward_kinematics import Link, DirectKinematic
from lib.frame import x_y_z_rotation_matrix, translation_matrix
from lib.utils import matrix_log6, inverse_transformation, se3_to_vec

np.set_printoptions(suppress=True, precision=6)

# change the links here
j0 = Link([0, 450, 150, np.pi / 2])
j1 = Link([np.pi / 2, 0, 590, 0])
j2 = Link([0, 0, 130, np.pi / 2])
j3 = Link([0, 647.07, 0, np.pi / 2])
j4 = Link([-np.pi / 2, 0, 0, -np.pi / 2])
j5 = Link([np.pi, 95, 0, 0])

home_offset = np.array([
  0, np.pi/2, 0, 0, -np.pi/2, np.pi
])

dk = DirectKinematic([j0, j1, j2, j3, j4, j5])


def ik(transformation_data=None, initial_guess=None, epsilon_wb=1e-5, epsilon_vb=1e-5, max_iterations=500, lmbd=.01,
       verbose=False):
  # transformation_data = [x, y, z, rx, ry, rz]
  # x, y, z: position of the end effector
  # rx, ry, rz: orientation of the end effector
  # returns: the joint angles

  # The end effector z-axis must be in the same direction and sign as the z-axis of the base frame

  n = dk.len_links

  if initial_guess is None:
    initial_guess = np.zeros(n)

  desired_rotation = x_y_z_rotation_matrix(transformation_data[3], transformation_data[4], transformation_data[5])
  desired_pose = translation_matrix(transformation_data[0], transformation_data[1],
                                    transformation_data[2]) @ desired_rotation

  theta_i = initial_guess

  error = True
  i = 0

  while error and i < max_iterations:
    htm = dk.get_htm(theta_i)
    i_htm = inverse_transformation(htm)

    Tbd = i_htm @ desired_pose
    log_tbd = matrix_log6(Tbd)

    s = se3_to_vec(log_tbd)

    J = dk.geometrical_jacobian(theta_i)

    d_theta = J.T @ np.linalg.inv(J @ J.T) @ s

    theta_i += (lmbd * d_theta)

    wb_err = np.linalg.norm(s[:3])
    vb_err = np.linalg.norm(s[3:])

    error = wb_err > epsilon_wb or vb_err > epsilon_vb

    i += 1

    if verbose:
      print(f'Iteration {i}, s = {s}')

  if verbose:
    print(f'Iterations: {i}')

  return theta_i, not error

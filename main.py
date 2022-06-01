#  Copyright (c) Miguel L. Rodrigues 2022.

import numpy as np

from lib.forward_kinematics import Link, DirectKinematic
from lib.frame import translation_matrix, x_y_z_rotation_matrix
from lib.utils import matrix_log6, inverse_transformation, se3_to_vec


np.set_printoptions(suppress=True, precision=6)


def normalize(r):
	return np.arctan2(
		np.sin(r),
		np.cos(r)
	)


# change the links here
j0 = Link([0, .45, .15, np.pi / 2])
j1 = Link([np.pi/2, 0, .59, 0])
j2 = Link([0, 0, .13, np.pi / 2])
j3 = Link([0, .64707, 0, -np.pi / 2])
j4 = Link([0, 0, 0, np.pi / 2])
j5 = Link([0, .095, 0, np.pi])

dk = DirectKinematic([j0, j1, j2, j3, j4, j5])


def ik(
	transformation_data,
	initial_guess=None,
	epsilon_wb=1e-5,
	epsilon_vb=1e-5,
	max_iterations=500,
	lmbd=1,
	method='least_squares',
	verbose=False):

	# transformation_data = [x, y, z, rx, ry, rz]
	# x, y, z: position of the end effector
	# rx, ry, rz: orientation of the end effector
	# returns: the joint angles
	
	# The end effector z-axis must be in the same direction and sign as the z-axis of the base frame

	n = dk.len_links
	
	if initial_guess is None:
		initial_guess = np.zeros(n)
	
	desired_pose = translation_matrix(transformation_data[0], transformation_data[1], transformation_data[2]) @ x_y_z_rotation_matrix(transformation_data[3], transformation_data[4], transformation_data[5])
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

		if method == 'pinv':
			grad = J.T @ np.linalg.inv(J @ J.T) @ s
		else:
			grad = J.T @ np.linalg.inv(J@J.T + (lmbd**2 * np.eye(n))) @ s

		theta_i += (lmbd * grad)

		wb_err = np.linalg.norm(s[:3])
		vb_err = np.linalg.norm(s[3:])
		
		error = wb_err > epsilon_wb or vb_err > epsilon_vb
		
		i += 1
	
	t = normalize(
		np.array(theta_i, dtype=np.float64)
	)
	
	if verbose:
		print(f'Iterations: {i}')
	
	return t, not error


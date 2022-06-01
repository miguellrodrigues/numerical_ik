import numpy as np
from numpy.linalg import norm

from lib.frame import z_rotation_matrix, translation_matrix, x_rotation_matrix


# Modern Robotics library


def near_zero(s):
	"""
		Returns True if the value is small enough to be considered zero.
		
	:param s: The value to check.
	"""
	return np.abs(s) < 1e-6


def normalize(v):
	"""
		Returns the normalized vector.
		
	:param v: The vector to normalize.
	:return: A unit vector in the same direction as v.
	"""
	return v / norm(v)


def inverse_rotation(R):
	"""
		Returns the inverse of a rotation matrix.
		
	:param R: The rotation matrix.
	:return: The inverse of rot.
	"""
	return R.T


def vec_to_so3(v):
	"""
		Converts a 3-vector to an so(3) representation
	:param v: A 3-vector
	:return: The skew symmetric representation of v
	"""
	return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def so3_to_vec(so3):
	"""
		Converts a so(3) representation to a 3-vector
	:param so3: A 3x3 skew-symmetric matrix
	:return: The 3-vector represented by so3
	"""
	return np.array([so3[2, 1], so3[0, 2], so3[1, 0]])


def exp3_to_axis_ang(exp3):
	"""
		Converts a 3-vector of exponential coordinates for rotation
		into axis-angle form
	:param exp3: A 3-vector of exponential coordinates for rotation
	:return: u: A unit rotation axis
	:return t: The rotation angle about u
	"""
	
	return normalize(exp3), norm(exp3)


def matrix_exp3(mat):
	"""
		Computes the matrix exponential of a matrix in so(3)
	:param mat: A 3x3 skew-symmetric matrix
	:return: The matrix exponential of mat
	"""
	
	v = so3_to_vec(mat)
	
	if near_zero(norm(v)):
		return np.eye(3)
	else:
		_, theta = exp3_to_axis_ang(v)
		s = mat / theta
		
		return np.eye(3) + np.sin(theta) * s + (1 - np.cos(theta)) * (s @ s)


def matrix_log3(R):
	"""
		Computes the matrix log of a rotation matrix
	:param R: A 3x3 rotation matrix
	:return: The matrix log of rot
	"""
	
	tr_r = (np.trace(R) - 1) / 2.0
	if tr_r >= 1:
		return 0, np.zeros((3, 3))
	elif tr_r <= -1:
		if not near_zero(1 + R[2][2]):
			s = (1.0 / np.sqrt(2 * (1 + R[2][2]))) * np.array([R[0][2], R[1][2], 1 + R[2][2]])
		elif not near_zero(1 + R[1][1]):
			s = (1.0 / np.sqrt(2 * (1 + R[1][1]))) * np.array([R[0][1], 1 + R[1][1], R[2][1]])
		else:
			s = (1.0 / np.sqrt(2 * (1 + R[0][0]))) * np.array([1 + R[0][0], R[1][0], R[2][0]])
		
		return np.pi, vec_to_so3(np.pi * s)
	else:
		theta = np.arccos(tr_r)
		return theta, theta / 2.0 / np.sin(theta) * (R - np.array(R).T)


def rotation_and_translation_to_transform(R, p):
	"""
		Converts a rotation matrix and a position vector into a homogeneous transformation
		matrix
	:param R: A 3x3 rotation matrix
	:param p: A 3-vector
	:return: A 4x4 homogeneous transformation matrix
	"""
	
	return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]


def transform_to_rotation_and_translation(T):
	"""
		Converts a homogeneous transformation matrix into a rotation matrix and a position vector
	:param T: A 4x4 homogeneous transformation matrix
	:return: A 3x3 rotation matrix and a 3-vector
	"""
	
	return T[:3, :3], T[:3, 3]


def inverse_transformation(T):
	"""
		Computes the inverse of a homogeneous transformation matrix
	:param T: A 4x4 homogeneous transformation matrix
	:return: The inverse of T
	"""
	
	R, p = transform_to_rotation_and_translation(T)
	R_inv = inverse_rotation(R)
	
	return rotation_and_translation_to_transform(R_inv, -(R_inv @ p))


def vec_to_se3(v):
	"""
		Converts a spatial velocity vector into a 4x4 matrix in se3
	:param v: A 6-vector representating a spatial velocity
	:return: A 4x4 matrix in se3
	"""
	
	return np.r_[np.c_[vec_to_so3(v[:3]), v[3:]], np.zeros((1, 4))]


def se3_to_vec(se3, reverse=False):
	"""
		Converts a 4x4 matrix in se3 into a 6-vector representing a spatial velocity
	:param reverse: If True, the spatial velocity is returned in the opposite direction
	:param se3: A 4x4 matrix in se3
	:return: A 6-vector representing a spatial velocity
	"""
	
	return np.r_[[se3[0][3], se3[1][3], se3[2][3]], [se3[2][1], se3[0][2], se3[1][0]]] if reverse else np.r_[
		[se3[2][1], se3[0][2], se3[1][0]], [se3[0][3], se3[1][3], se3[2][3]]]


def adjoint(T):
	"""
		Computes the adjoint representation of a homogeneous transformation matrix
	:param T: A Homogeneoous transformation matrix
	:return: The 6x6 adjoint representation of T
	"""
	
	R, p = transform_to_rotation_and_translation(T)
	return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[vec_to_so3(p).dot(R), R]]


def screw_to_axis(q, s, h):
	"""
		Converts a parametric description of a screw axis and coverts it into an normalized
		screw axis
	:param q: A point on the screw axis
	:param s: A unit vector in the direction of the screw axis
	:param h: The pitch of the screw axis
	:return: A normalized screw axis
	"""
	
	return np.r_[s, np.cross(q, s) + (h * s)]


def exp6_to_axis_ang(exp6):
	"""
		Converts a 6-vector of exponential coordinates into screw axis-angle
	:param exp6: A 6-vector of exponential coordinates for rigid-body motion S*theta
	:return: S: The corresponding normalized screw axis
	return: theta: The distance traveled along S
	"""
	
	theta = norm(exp6[:3])
	if near_zero(theta):
		theta = norm(exp6[3:])
	
	return np.array(exp6 / theta), theta


def matrix_exp6(mat):
	"""
		Computes the matrix exponential of a se3 representation of exponential coordinates
	:param mat: A matrix in se3
	:return: The matrix exponential of se3mat
	"""
	
	s = so3_to_vec(mat[:3, :3])
	if near_zero(norm(s)):
		return np.r_[np.c_[np.eye(3), mat[:3, 3]], [[0, 0, 0, 1]]]
	else:
		theta = exp3_to_axis_ang(s)[1]
		u = mat[:3, :3] / theta
		
		return np.r_[np.c_[u, np.dot(np.eye(3) * theta + (1 - np.cos(theta)) * u + (theta - np.sin(theta)) * np.dot(u, u),
																 mat[:3, 3]) / theta], [[0, 0, 0, 1]]]


def matrix_log6(T):
	"""
	Computes the matrix log of a homogeneous transformation matrix
	:param T: A matrix in SE3
	:return: The matrix log of T
	"""
	
	R, p = transform_to_rotation_and_translation(T)
	_, l3 = matrix_log3(R)
	
	if np.array_equal(l3, np.zeros((3, 3))):
		return np.r_[np.c_[np.zeros((3, 3)), [T[0][3], T[1][3], T[2][3]]], [[0, 0, 0, 0]]]
	else:
		theta = np.arccos((np.trace(R) - 1) / 2.0)
		
		return np.r_[np.c_[l3, np.dot(
			np.eye(3) - l3 / 2.0 + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) * np.dot(l3, l3) / theta,
			[T[0][3], T[1][3], T[2][3]])], [[0, 0, 0, 0]]]


def compute_link_transformation(dhp):
	rz = z_rotation_matrix(dhp[0])
	tz = translation_matrix(0, 0, dhp[1])
	tx = translation_matrix(dhp[2], 0, 0)
	rx = x_rotation_matrix(dhp[3])
	
	return rz @ tz @ tx @ rx


def compute_homogeneous_transformation(links, start, end, joint_angles):
	if end == 0:
		return np.eye(4)
	
	tm = links[start].get_tm(joint_angles[0])
	
	for i in range(start + 1, end):
		tm_i = links[i].get_tm(joint_angles[i])
		tm = tm @ tm_i
	
	return tm

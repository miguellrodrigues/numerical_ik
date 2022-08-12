#  Copyright (c) 2022.

import numpy as np


def x_rotation_matrix(yaw):
	"""
	Rotation matrix around the x axis
	"""
	return np.array([[1, 0, 0, 0], [0, np.cos(yaw), -np.sin(yaw), 0], [0, np.sin(yaw), np.cos(yaw), 0], [0, 0, 0, 1]])


def y_rotation_matrix(pitch):
	"""
	Rotation matrix around the y axis
	"""
	return np.array(
		[[np.cos(pitch), 0, np.sin(pitch), 0], [0, 1, 0, 0], [-np.sin(pitch), 0, np.cos(pitch), 0], [0, 0, 0, 1]])


def z_rotation_matrix(roll):
	"""
	Rotation matrix around the z axis
	"""
	return np.array([[np.cos(roll), -np.sin(roll), 0, 0], [np.sin(roll), np.cos(roll), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def x_y_z_rotation_matrix(yaw, pitch, roll):
	"""
	Rotation matrix around the x, y and z axis
	"""
	return x_rotation_matrix(yaw) @ y_rotation_matrix(pitch) @ z_rotation_matrix(roll)


def arbitrary_vector_rotation_matrix(theta, v):
	"""
	Rotation matrix around an arbitrary vector
	"""
	
	u = v
	su = np.array([[0, -u[2], u[1], 0], [u[2], 0, -u[0], 0], [-u[1], u[0], 0, 0], [0, 0, 0, 1]])
	
	return np.eye(4) + np.sin(theta) * su + (1 - np.cos(theta)) * (su @ su)


def translation_matrix(dx, dy, dz):
	return np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])


class Frame:
	def __init__(self, x, y, z, yaw=0, pitch=0, roll=0, name=''):
		self.position = translation_matrix(x, y, z)
		
		self.rotation = x_y_z_rotation_matrix(yaw, pitch, roll)
		
		self.name = name
	
	def translate(self, dx, dy, dz):
		self.position = translation_matrix(dx, dy, dz) @ self.position
		
		return self.position
	
	def rotate(self, yaw, pitch, roll):
		self.rotation = x_y_z_rotation_matrix(yaw, pitch, roll) @ self.rotation
	
	def rotate_around_arbitrary_vector(self, theta, v):
		self.rotation = arbitrary_vector_rotation_matrix(theta, v) @ self.rotation
	
	def get_x_component(self):
		return self.position[0, 3]
	
	def get_y_component(self):
		return self.position[1, 3]
	
	def get_z_component(self):
		return self.position[2, 3]
	
	def rotation_matrix(self):
		return self.rotation
	
	def rotation_to(self, other):
		yaw = np.arctan2(other.rotation[2, 1], other.rotation[2, 2]) - np.arctan2(self.rotation[2, 1], self.rotation[2, 2])
		pitch = np.arctan2(other.rotation[2, 0], other.rotation[2, 2]) - np.arctan2(self.rotation[2, 0],
																																								self.rotation[2, 2])
		roll = np.arctan2(other.rotation[1, 0], other.rotation[0, 0]) - np.arctan2(self.rotation[1, 0], self.rotation[0, 0])
		
		return np.array([yaw, pitch, roll])

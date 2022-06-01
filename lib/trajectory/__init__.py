import numpy as np


class Trajectory:
	def __init__(self, restrictions, d_time):
		self.restrictions = restrictions
		self.d_time = d_time
		
		self.temporal_matrix = self.compute_temporal_matrix()
		self.coefficients = np.linalg.inv(self.temporal_matrix) @ self.restrictions
		
	def compute_temporal_matrix(self):
		m = np.array([
			[1, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0],
			[0, 0, 2, 0, 0, 0],
			[1, self.d_time, self.d_time ** 2, self.d_time ** 3, self.d_time ** 4, self.d_time ** 5],
			[0, 1, 2 * self.d_time, 3 * self.d_time ** 2, 4 * self.d_time ** 3, 5 * self.d_time ** 4],
			[0, 0, 2, 6 * self.d_time, 12 * self.d_time ** 2, 20 * self.d_time ** 3],
		])
		
		return m
	
	def get_trajectory(self, t):
		return self.coefficients @ np.array([1, t, t ** 2, t ** 3, t ** 4, t ** 5])
	
	def get_velocity(self, t):
		return self.coefficients @ np.array([0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4])
	
	def get_acceleration(self, t):
		return self.coefficients @ np.array([0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3])

#  Copyright (c) 2022.

import numpy as np

from lib.utils import compute_link_transformation, compute_homogeneous_transformation


class Link:

  def __init__(self, dhp, link_type='rotational'):
    self.dhp = dhp
    self.link_type = link_type
    self.transformation_matrix = compute_link_transformation(dhp)

  def get_tm(self, joint_angle=None):
    if joint_angle is not None:
      self.update_tm(joint_angle)

    return self.transformation_matrix

  def update_tm(self, theta):
    self.transformation_matrix = compute_link_transformation(
      [self.dhp[0] + theta,
       self.dhp[1],
       self.dhp[2],
       self.dhp[3]]
    )


class DirectKinematic:
  def __init__(self, links):
    self.links = links
    self.len_links = len(links)

  def get_transformation(self, start, end, joint_angles=None):
    tf = compute_homogeneous_transformation(self.links, start, end, joint_angles)
    return tf

  def get_htm(self, joint_angles):
    return compute_homogeneous_transformation(self.links, 0, len(self.links), joint_angles)

  def geometrical_jacobian(self, joint_angles):
    htm = self.get_htm(joint_angles)

    len_joints = len(self.links)
    j = np.zeros((6, len_joints))

    # J_pi = Z_i-1 x (P - pi-1)
    # J_oi = z_i-1

    P = htm[:3, 3]

    p_i = np.array([0, 0, 0])
    z_i = np.array([0, 0, 1])

    for i in range(1, len_joints + 1):
      p_diff = (P - p_i)

      J_pi = np.cross(z_i, p_diff)
      J_oi = z_i

      J = np.hstack((J_oi, J_pi))
      j[:, i - 1] = J

      transformation = self.get_transformation(0, i, joint_angles)

      p_i = transformation[:3, 3]
      z_i = transformation[:3, 2]

    return j

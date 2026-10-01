"""
Kinematics for the UFactory Lite 6.

TODO: Implement all functions marked with TODO below.
      For FK, implement either the DH method or the PoX method (your choice).
      For IK, implement IK_geometric.
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


# ======================================================================
# FK - DH Method
# ======================================================================

# Standard DH parameters for the UFactory Lite 6: https://docs.supportarticle.ufactory.cc/support_articles/developer/kinematic-and-dynamic-parameters/lite6.html
# One row per joint: [theta_offset (rad), d (mm), alpha (rad), a (mm)]
DH_STD = np.array([
    [0.0, 243.3, np.deg2rad(-90.0), 0.0], # TODO: student lab
    [np.deg2rad(-90.0), 0.0, np.deg2rad(180.0), 200.0],
    [np.deg2rad(-90.0), 0.0, np.deg2rad(90.0), 87.0],
    [0.0, 227.6, np.deg2rad(90.0), 0.0],
    [0.0, 0.0, np.deg2rad(-90.0), 0.0],
    [0.0, 61.5, 0.0, 0.0],
], dtype=float)

# Lite 6 joint travel limits, matching the MuJoCo model
# (robot_assets/UfactoryLite6/xml/lite6.xml). Joint 3 is asymmetric.
JOINT_LIMITS_DEG = np.array([
    [-360.0, 360.0],   # Joint 1
    [-150.0, 150.0],   # Joint 2
    [  -3.5, 300.0],   # Joint 3
    [-360.0, 360.0],   # Joint 4
    [-124.0, 124.0],   # Joint 5
    [-360.0, 360.0],   # Joint 6
], dtype=float)
JOINT_LIMITS = np.radians(JOINT_LIMITS_DEG)   # (6, 2) in radians

# The arm's default (home) configuration, matching the UFactory Studio default.
# This is both the pose Initial Pose drives to and the seed IK_numerical starts from.
Q_DEFAULT_DEG = np.array([0.0, 9.9, 31.8, 0.0, 21.9, 0.0], dtype=float)
Q_DEFAULT = np.radians(Q_DEFAULT_DEG)         # (6,) in radians


def get_transform_from_dh(theta_offset, d, alpha, a, joint_angle):
    """
    Build the 4x4 transform for one DH row.

    DH row format (from arm.dh_params):
        theta_offset (rad), d (mm), alpha (rad), a (mm)
    joint_angle: current joint angle in radians.
    """

    theta = theta_offset + joint_angle

    rot_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(alpha),-np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0,1 ],
        ])
    rot_z = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
    trans_x = np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
    trans_z = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1],
        ])
    
    return rot_z @ trans_z @ trans_x @ rot_x


def FK_dh(dh_params, joint_angles_rad, num_joints):
    """
    Forward kinematics via DH convention.

    Calls get_transform_from_dh for each joint and chains the results.

    dh_params:        DH table, flat list (SDK arm.dh_params has 7 rows, 4 values) or a (n, 4) array.
                      Each row: [theta_offset (rad), d (mm), alpha (rad), a (mm)]
    joint_angles_rad: joint angles in radians, length num_joints.
    num_joints:       number of joints to chain (pass 6 for end-effector).

    Returns 4x4 homogeneous transform (base -> end-effector).
    """
    
    H = np.eye(4)
    for i, (theta, d, alpha, a) in enumerate(dh_params):

        # breakpoint()
        current_joint_angle = joint_angles_rad[i]
        H_temp = get_transform_from_dh(theta, d, alpha, a, current_joint_angle)
        H = H @ H_temp

    return H




# ======================================================================
# FK - PoX Method
# ======================================================================

# M and S_list are robot constants for the PoX method:
#   M:      4x4 end-effector transform at the zero (home) configuration.
#   S_list: 6x6 - one space-frame screw axis per row: [w1, w2, w3, v1, v2, v3]
# Units are mm, like FK_dh.

M = np.array([
    [1, 0, 0, 87],  # TODO: student lab
    [0, -1, 0, 0],
    [0, 0, -1, 154.2],
    [0, 0, 0, 1],
], dtype=float)

S_list = np.array([
    [0, 0, 1, 0, 0, 0],  # TODO: student lab
    [0, 1, 0, -243.3, 0, 0],
    [0, -1, 0, 443.3, 0, 0],
    [0, 0, -1, 0, 87, 0],
    [0, 1, 0, -215.7, 0, 87],
    [0, 0, -1, 0, 87, 0],
], dtype=float)


def to_s_matrix(w, v, theta):
    """
    Build the 4x4 [S] skew-symmetric matrix for a screw axis.
    w: (3,) angular velocity component
    v: (3,) linear velocity component
    """
    # TODO: student lab
    skew_symmetric = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
    ])

    R = np.eye(3) + np.sin(theta) * skew_symmetric + (1 -np.cos(theta)) * (skew_symmetric @ skew_symmetric)

    p = ((np.eye(3) * theta) @ v + (1 - np.cos(theta)) * (skew_symmetric @ v) + (theta - np.sin(theta)) * (skew_symmetric @ (skew_symmetric @ v))).reshape(-1, 1)

    e_S_theta = np.vstack((np.hstack((R, p)), np.array([0,0,0,1])))

    return e_S_theta
    


def FK_pox(joint_angles_rad, m_mat, s_lst):
    """
    Forward kinematics via product of exponentials.

    Calls to_s_matrix for each joint and chains e^([S]*theta).

    joint_angles_rad: joint angles in radians (raw angles, offsets live in M).
    m_mat:            4x4 home configuration matrix M.
    s_lst:            (6, 6) screw axes, one per row: [w1, w2, w3, v1, v2, v3].

    Returns 4x4 homogeneous transform (base -> end-effector).
    """
    H = np.eye(4)
    for i, s in enumerate(s_lst):
        H = H @ to_s_matrix(s[:3], s[3:], joint_angles_rad[i])
    return H @ m_mat

# ======================================================================
# FK - shared output helper
# ======================================================================

def get_pose_from_T(T):
    """
    Extract [x, y, z, phi, theta, psi] from a 4x4 homogeneous transform.
    x, y, z  in mm.  phi, theta, psi  in radians.

    Orientation is roll-pitch-yaw about fixed X, Y, Z axes (matches the
    xArm SDK convention and the GUI's Roll/Pitch/Yaw readout).

    Used by both FK_dh and FK_pox to produce the pose vector for the GUI.
    """

    Rotation_mat = T[:3, :3]
    translation_vec = T[:3, 3]

    rot = R.from_matrix(Rotation_mat)
    roll, pitch, yaw = rot.as_euler('xyz')

    return [translation_vec[0], translation_vec[1], translation_vec[2], roll, pitch, yaw]


# ======================================================================
# IK - Geometric Method
# ======================================================================

def IK_geometric(dh_params, pose):
    """
    Inverse kinematics for the Lite 6.

    pose: [x, y, z, phi, theta, psi]  (mm, radians).

    Returns joint angles in radians as a numpy array of length 6,
    or None if the pose is not reachable.
    """
    # TODO: student lab

    x, y, z, phi, theta, psi = pose

    # link lengths (mm)
    l1 = 243,3
    l2 = 200
    l3 = (87**2 + 227.6**2)**0.5
    l6 = 61.5

    target_end_effector_pos = np.array([x, y, z])

    # Step 1: Find wrist center

    R_z_phi = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1],
    ])

    R_y_theta = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])

    R_x_psi = np.array([
        [1, 0, 0],
        [0, np.cos(psi),-np.sin(psi)],
        [0, np.sin(psi), np.cos(psi)],
    ])

    R_0_6 = R_z_phi @ R_y_theta @ R_x_psi

    o_c = target_end_effector_pos - l6 * R_0_6 @ np.array([0, 0, 1])
    x_c, y_c, z_c = o_c

    # Step 2: Solve joint 1
    theta_1 = np.atan2(y, x)

    # Step 3: Solve joints 2 & 3
    r = (x_c**2 + y_c**2)**0.5
    s = (z_c**2 - l1**2)**0.5
    theta_3 = np.arccos(((x_c**2 + y_c**2) - l2**2 - l3**2) / (2 * l2 * l3))
    if np.isnan(theta_3):
        return None, "The pose is not reachable: theta_3 fails"
    if theta_3 < JOINT_LIMITS[2][0] or theta_3 > JOINT_LIMITS[2][1]:
        theta_3 = -theta_3
        if theta_3 < JOINT_LIMITS[2][0] or theta_3 > JOINT_LIMITS[2][1]:
            return None, "The pose is not reachable: theta_3 fails"
    
    theta_2 = np.atan2(y_c, x_c) - np.atan2(l2 * np.sin(theta_3), l2 + l3 * np.cos(theta_3))
    if theta_2 < JOINT_LIMITS[1][0] or theta_2 > JOINT_LIMITS[1][1]:
        return None, "The pose is not reachable: theta_2 fails"

    # Step 4:  Solve joints 4, 5, & 6
    R_0_3 = np.array([
        [np.cos(theta_1) * np.cos(theta_2 + theta_3), -np.cos(theta_1) * np.sin(theta_2 + theta_3), np.sin(theta_1)],
        [np.sin(theta_1) * np.cos(theta_2 + theta_3), -np.sin(theta_1) * np.sin(theta_2 + theta_3), -np.cos(theta_1)],
        [np.sin(theta_2 + theta_3), np.cos(theta_2 + theta_3), 0],
    ])

    R_0_3_inv = np.linalg.inv(R_0_3)
    R_3_6 = R_0_3_inv @ R_0_6

    r_1_3 = R_3_6[0][2]
    r_2_3 = R_3_6[1][2]
    r_3_3 = R_3_6[2][2]
    r_3_2 = R_3_6[2][1]
    r_3_1 = R_3_6[2][0]
    r_2_1 = R_3_6[1][0]
    r_1_1 = R_3_6[0][0]

    # Case 2
    if  r_1_3 == 0 and r_2_3 == 0:  # theta_5 = N*pi, theta_5 in [-124, 124]
        if r_3_3 == 1:
            theta_5 = 0
            theta_6 = 0
            theta_4 = np.atan2(r_2_1, r_1_1)
        if r_3_3 == -1:
            return None, "The pose is not reachable: theta_5 check fails in Case 2"

    # Case 1
    else:

        pos_sin_theta = (1 - r_3_3**2)**0.5
        neg_sin_theta = -pos_sin_theta

        theta_5_neg = np.atan2(neg_sin_theta, r_3_3)
        theta_5_pos = np.atan2(pos_sin_theta, r_3_3)

        # check theta cases
        if theta_5_pos <= JOINT_LIMITS[4][1] and theta_5_pos >= JOINT_LIMITS[4][0]:
            theta_4 = np.atan2(r_2_3, r_1_3)
            theta_5 = theta_5_pos
            theta_6 = np.atan2(r_3_2, -r_3_1)

        elif theta_5_neg >= JOINT_LIMITS[4][0] and theta_5_neg >= JOINT_LIMITS[4][0]:
            theta_4 = np.atan2(-r_2_3, -r_1_3)
            theta_5 = theta_5_neg
            theta_6 = np.atan2(-r_3_2, r_3_1)

        else:
            return None, "The pose is not reachable: fails theta_5 in Case 1"

    joint_angles = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])

    H_0_6 = FK_pox(joint_angles, M, S_list)

    FK_pos = H_0_6[:3, -1]
    pos_err = np.linalg.norm(FK_pos - pose[:3])

    R_FK = H_0_6[:3, :3]
    R_err = R_FK @ R_0_6.T
    angle_err = np.arcos(np.clip(np.trace(R_err) - 1) / 2, -1, -1)

    if pos_err >= 1 or angle_err >= 0.1:
        return None, "The pose is not reachable: doesn't pass FK check"

    return joint_angles


# ======================================================================
# IK - Numerical Method (bounded Gauss-Newton)
# ======================================================================

def IK_numerical(dh_params, pose, q0=None, joint_limits=None, w_rot=200.0):
    """
    Inverse kinematics for the Lite 6 by bounded Gauss-Newton least squares.

    Solves  min_q ||W e(q)||^2  s.t.  lb <= q <= ub, where e is the pose error
    between FK_dh(q) and the requested pose.

    pose:         [x, y, z, roll, pitch, yaw]  (mm, radians) - same as IK_geometric.
    q0:           starting guess in radians. None seeds from Q_DEFAULT (the home
                  pose) rather than the arm's current angles, so repeated calls
                  are reproducible.
    joint_limits: (6, 2) array of [lo, hi] in radians; defaults to JOINT_LIMITS.
    w_rot:        mm per radian, putting the position and rotation blocks of the
                  residual on one scale. Only matters when a limit binds and the
                  pose cannot be reached exactly.

    Returns joint angles in radians as a numpy array of length 6,
    or None if the pose is not reachable.
    """
    # TODO: student lab
    return None


# ======================================================================
# Standalone test
# ======================================================================

if __name__ == '__main__':
    # get_transform_from_dh with all zeros should return identity
    T = get_transform_from_dh(0, 0, 0, 0, 0)
    if T is not None:
        print("get_transform_from_dh(all zeros):")
        print(T)
    else:
        print("get_transform_from_dh not yet implemented")

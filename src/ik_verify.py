import random
from kinematics import *
import math


def get_joint_vectors(size):
    joint_limits = []
    for i in range(0, size):
        vector = []

        joint_1 = np.radians(random.uniform(-360, 360))
        joint_2 = np.radians(random.uniform(-150, 150))
        joint_3 = np.radians(random.uniform(-3.5, 300))
        joint_4 = np.radians(random.uniform(-360, 360))
        joint_5 = np.radians(random.uniform(-124, 124))
        joint_6 = np.radians(random.uniform(-360, 360))

        vector.extend([joint_1, joint_2, joint_3,
                             joint_4, joint_5, joint_6])

        joint_limits.append(vector)

    return joint_limits


def orientation_error(R_a, R_b):
    """Angle of the rotation taking R_b to R_a, in radians."""
    return np.linalg.norm(R.from_matrix(R_a @ R_b.T).as_rotvec())


def pose_solver(dh_params, size):

    joint_limits = get_joint_vectors(size)

    errors_pos_geom = []
    errors_pos_num = []

    errors_orientation_geom = []
    errors_orientation_num = []

    failures_geom = 0
    failures_num = 0

    for joint_vector in joint_limits:

        fk_matrix = FK_pox(joint_vector, M, S_list)

        # convert to pose from homogenous matrix
        fk_pose = get_pose_from_T(fk_matrix)

        ik_geom = IK_geometric(dh_params, fk_pose)
        ik_num = IK_numerical(dh_params, fk_pose)

        # IK Geometric
        if ik_geom is None:
            failures_geom += 1
        else:
            final_geom = FK_pox(ik_geom, M, S_list)
            errors_pos_geom.append(np.linalg.norm(final_geom[:3, 3] - fk_matrix[:3, 3]))
            errors_orientation_geom.append(orientation_error(final_geom[:3, :3], fk_matrix[:3, :3]))

        # IK Numerical
        if ik_num is None:
            failures_num += 1
        else:
            final_num = FK_pox(ik_num, M, S_list)
            errors_pos_num.append(np.linalg.norm(final_num[:3, 3] - fk_matrix[:3, 3]))
            errors_orientation_num.append(orientation_error(final_num[:3, :3], fk_matrix[:3, :3]))

    print(f"Geometric IK solved {size - failures_geom}/{size} poses")
    print(f"Numerical IK solved {size - failures_num}/{size} poses")
    print("\n")
    print("Position Errors")
    print(f"Mean Position Error for Geometric IK solver: {np.mean(errors_pos_geom)} mm")
    print(f"Max Position Error for Geometric IK solver: {np.max(errors_pos_geom)} mm")
    print(f"Mean Position Error for Numerical IK sovler: {np.mean(errors_pos_num)} mm")
    print(f"Max Position Error for Numerical IK solver: {np.max(errors_pos_num)} mm")
    print("\n")
    print("Orientation Errors")
    print(f"Mean Orientation Error for Geometric IK solver: {np.mean(errors_orientation_geom)} rad")
    print(f"Max Orientation Error for Geometric IK solver: {np.max(errors_orientation_geom)} rad")
    print(f"Mean Orientation Error for Numerical IK sovler: {np.mean(errors_orientation_num)} rad")
    print(f"Max Orientation Error for Numerical IK solver: {np.max(errors_orientation_num)} rad")

if __name__ == "__main__":
    pose_solver(DH_STD, 500)

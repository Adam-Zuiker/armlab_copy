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


def pose_solver(dh_params, size):

    joint_limits = get_joint_vectors(size)

    errors_pos_geom = []
    errors_pos_num = []

    errors_orientation_geom = []
    errors_orientation_num = []

    for joint_vector in joint_limits:

        fk_matrix = FK_pox(joint_vector, M, S_list)

        # convert to pose from homogenous matrix
        fk_pose = get_pose_from_T(fk_matrix)

        ik_geom = IK_geometric(dh_params, fk_pose)
        ik_num = IK_numerical(dh_params, fk_pose)

        final_pose_geom = get_pose_from_T(FK_pox(ik_geom))
        final_pose_num = get_pose_from_T(FK_pox(ik_num))

        # position error
        pos_final_geom = final_pose_geom[0:3]
        error_pos_geom = np.linalg.norm(pos_final_geom - fk_pose[0:3])

        errors_pos_geom.append(error_pos_geom)

        pos_final_num = final_pose_num[0:3]
        error_pos_num = np.linalg.norm(pos_final_num - fk_pose[0:3])

        errors_pos_num.append(error_pos_num)

        # orientation error - IK Geometric
        rotation = R.from_euler('zyx', [final_pose_geom[5], final_pose_geom[4], final_pose_geom[3]], degrees = False)
        rotation_matrix = rotation.as_matrix()
        rot_error = rotation_matrix @ fk_pose[0:3, 0:3].T
        rotvec_error_geom = np.linalg.norm(rot_error.as_rotvec())

        errors_orientation_geom.append(rotvec_error_geom)

        # orientation error - IK Numerical
        rotation = R.from_euler('zyx', [final_pose_geom[5], final_pose_geom[4], final_pose_geom[3]], degrees = False)
        rotation_matrix = rotation.as_matrix()
        rot_error = rotation_matrix @ fk_pose[0:3, 0:3].T
        rotvec_error_geom = np.linalg.norm(rot_error.as_rotvec())

        errors_orientation_geom.append(rotvec_error_geom)

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


    

    




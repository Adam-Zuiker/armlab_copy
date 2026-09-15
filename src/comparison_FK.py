
from lite6arm import Lite6Arm
from kinematics import JOINT_LIMITS, get_pose_from_T, FK_dh, DH_STD
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

def compare_FK():
    arm = Lite6Arm()

    differences = JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]

    joint_angles = []
    euclidean_errors = []
    angle_errors = []

    for i in range(500): 
        rands = np.random.rand(1, 6)

        
        offset = rands * differences
        q = offset + JOINT_LIMITS[:, 0]
        q = q.squeeze()

        joint_angles.append(q)


        code, pose_sdk = arm.xarm.get_forward_kinematics(q)

        H = FK_dh(DH_STD, q, 6)

        pose_sdk = np.array(pose_sdk)

        our_pose = np.array(get_pose_from_T(H))

        pose_diff_euclidean = np.linalg.norm(pose_sdk[:3] - our_pose[:3])

        euclidean_errors.append(pose_diff_euclidean)

        pose_sdk_rotation = R.from_euler('xyz', pose_sdk[3:], degrees=False)
        pose_sdk_rotation_mat = pose_sdk_rotation.as_matrix()

        our_pose_rotation = R.from_euler('xyz', our_pose[3:], degrees=False)
        our_pose_rotation_mat = our_pose_rotation.as_matrix()

        R_error = our_pose_rotation_mat.T @ pose_sdk_rotation_mat

        angle = np.arccos((np.trace(R_error) - 1) / 2)

        angle_errors.append(angle)

    mean_euclidian = np.mean(euclidean_errors)
    mean_angles = np.mean(angle_errors)
    median_euclidian = np.median(euclidean_errors)
    median_angles = np.median(angle_errors)
    max_euclidian = np.max(euclidean_errors)
    max_angles = np.max(angle_errors)
    p95_euclidian = np.percentile(euclidean_errors, 95)
    p95_angles = np.percentile(angle_errors, 95)

    fig, (plt1, plt2) = plt.subplots(1, 2, figsize = (5, 5))
    plt1.hist(euclidean_errors, bins = 50, color = 'blue')
    plt1.set_title('Histogram of Errors, Euclidian Distance')
    plt1.set_xlabel('Position Error')
    plt1.set_ylabel('Frequency')

    plt2.hist(angle_errors, bins = 50, color = 'red')
    plt2.set_title("Histogram of Errors, Joint Angles")
    plt2.set_xlabel("Degree Error")
    plt2.set_ylabel("Frequency")
    fig.show()

    joint_angles = np.array(joint_angles)


    fig1, joint_axes = plt.subplots(3, 2, figsize = (16, 12))
    

    for i, ax in enumerate(joint_axes.flatten()):
        ax.scatter(joint_angles[:, i], angle_errors)
        ax.set_xlabel(f"Joint {i+1} Angle")
        ax.set_ylabel("Angle Error")
        ax.set_title(f"Joint Angle vs Angle Error (Joint {i+1})")

    

   

    fig2, joint_axes2 = plt.subplots(3, 2, figsize = (16, 12))
    for i, ax in enumerate(joint_axes2.flatten()):
        ax.scatter(joint_angles[:, i], euclidean_errors)
        ax.set_xlabel(f"Joint {i+1} Angle")
        ax.set_ylabel("Angle Error")
        ax.set_title(f"Joint Angle vs Eudlidean Error (Joint {i+1})")
        
    

    fig1.show()
    fig2.show()







    breakpoint()

if __name__ == "__main__":
    compare_FK()
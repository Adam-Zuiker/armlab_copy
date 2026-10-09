import random
from kinematics import *
# from lite6arm import Lite6Arm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


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


def pose_solver(dh_params, size, arm):

    joint_limits = get_joint_vectors(size)

    errors_pos_geom = []
    errors_pos_num = []
    errors_pos_vendor = []

    errors_orientation_geom = []
    errors_orientation_num = []
    errors_orientation_vendor = []

    failures_geom = 0
    failures_num = 0
    failures_vendor = 0

    for joint_vector in joint_limits:

        fk_matrix = FK_pox(joint_vector, M, S_list)

        # convert to pose from homogenous matrix
        fk_pose = get_pose_from_T(fk_matrix)

        ik_geom = IK_geometric(dh_params, fk_pose)
        ik_num = IK_numerical(dh_params, fk_pose)
        ik_vendor = arm.xarm.get_inverse_kinematics(fk_pose)[1]

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

        # IK Vendor
        if ik_vendor is None:
            failures_vendor += 1
        else:
            final_vendor = FK_pox(ik_vendor, M, S_list)
            errors_pos_vendor.append(np.linalg.norm(final_vendor[:3, 3] - fk_matrix[:3, 3]))
            errors_orientation_vendor.append(orientation_error(final_vendor[:3, :3], fk_matrix[:3, :3]))

    print(f"Geometric IK solved {size - failures_geom}/{size} poses")
    print(f"Numerical IK solved {size - failures_num}/{size} poses")
    print(f"Vendor IK solved {size - failures_vendor}/{size} poses")
    print("\n")
    print("Position Errors")
    print(f"Mean Position Error for Geometric IK solver: {np.mean(errors_pos_geom)} mm")
    print(f"Max Position Error for Geometric IK solver: {np.max(errors_pos_geom)} mm")
    print(f"Mean Position Error for Numerical IK sovler: {np.mean(errors_pos_num)} mm")
    print(f"Max Position Error for Numerical IK solver: {np.max(errors_pos_num)} mm")
    print(f"Mean Position Error for Vendor IK sovler: {np.mean(errors_pos_vendor)} mm")
    print(f"Max Position Error for Vendor IK solver: {np.max(errors_pos_vendor)} mm")
    print("\n")
    print("Orientation Errors")
    print(f"Mean Orientation Error for Geometric IK solver: {np.mean(errors_orientation_geom)} rad")
    print(f"Max Orientation Error for Geometric IK solver: {np.max(errors_orientation_geom)} rad")
    print(f"Mean Orientation Error for Numerical IK sovler: {np.mean(errors_orientation_num)} rad")
    print(f"Max Orientation Error for Numerical IK solver: {np.max(errors_orientation_num)} rad")
    print(f"Mean Orientation Error for Vendor IK sovler: {np.mean(errors_orientation_vendor)} rad")
    print(f"Max Orientation Error for Vendor IK solver: {np.max(errors_orientation_vendor)} rad")

def create_reachability_map(dh_params, block_size):
    # define flange pose [0, 0, z, phi, theta, psi] for 38 mm block
    reachable = {}
    z = 85 + block_size / 2
    # sweep across x & y on board
    for x in range(0, 500, 10):
        for y in range(-500, 500, 10):
            pose_0 = [x, y, z, np.pi, 0, 0]
            ik_geom_0 = IK_geometric(dh_params, pose_0)
            pose_90 = [x, y, z, np.pi, 0, np.pi/2]
            ik_geom_90 = IK_geometric(dh_params, pose_90)

            if ik_geom_0 is not None and ik_geom_90 is not None:
                reachable[(x, y)] = True
            else:
                reachable[(x, y)] = False
    plot_reachability(reachable, block_size)

def plot_reachability(reachable, block_size):
    x_unique = sorted(list(set(k[0] for k in reachable.keys())))
    y_unique = sorted(list(set(k[1] for k in reachable.keys())))    

    x_to_idx = {x: i for i, x in enumerate(x_unique)}
    y_to_idx = {y: i for i, y in enumerate(y_unique)}

    # breakpoint()

    Z_grid = np.zeros((len(y_unique), len(x_unique)), dtype=bool)

    for (x, y), value in reachable.items():
        row = y_to_idx[y]
        col = x_to_idx[x]
        Z_grid[row, col] = value

    plt.figure(figsize=(6, 6))
    plt.pcolormesh(x_unique, y_unique, Z_grid, cmap='viridis', shading='auto')

    plt.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)

    base_size = 120  # mm - TODO: measure the base footprint on your station
    plt.gca().add_patch(Rectangle(
        (-base_size / 2 + 60, -base_size / 2),   # bottom-left corner, so the square is centered on (0, 0)
        base_size, base_size,
        facecolor='gray', alpha=0.6, edgecolor='black', hatch='//', label='Arm base',
    ))
    plt.legend(loc='upper right')
    plt.gca().set_aspect('equal') 

    # Display the plot
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Condition Grid (True = Yellow, False = Purple)')
    plt.savefig(f'./screenshot/reachability_map_{block_size}mm_block.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # arm = Lite6Arm()
    # pose_solver(DH_STD, 500, arm)
    create_reachability_map(DH_STD, 38)
    create_reachability_map(DH_STD, 25)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from kinematics import DH_STD, JOINT_LIMITS, M, S_list, FK_dh, FK_pox
from lite6arm import Lite6Arm


NUM_SAMPLES = 500
RNG_SEED = 550


def rotation_error(R_a, R_b):
    """Return the smallest rotation from R_a to R_b, in radians."""
    R_error = R_a.T @ R_b
    cos_angle = (np.trace(R_error) - 1.0) / 2.0
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def print_error_statistics(label, position_errors, orientation_errors):
    """Print the required FK comparison statistics."""
    print(f"\n{label}")
    for name, values, unit in (
        ("Euclidean distance error", position_errors, "mm"),
        ("Orientation error", orientation_errors, "rad"),
    ):
        print(f"  Mean {name}: {np.mean(values):.12g} {unit}")
        print(f"  Median {name}: {np.median(values):.12g} {unit}")
        print(f"  Maximum {name}: {np.max(values):.12g} {unit}")
        print(f"  95th percentile {name}: {np.percentile(values, 95):.12g} {unit}")


def save_figure(figure, filename):
    output_dir = Path(__file__).resolve().parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / filename
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    print(f"Saved plot: {path}")


def plot_histograms(title, position_errors, orientation_errors):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(position_errors, bins=50, color="tab:blue")
    axes[0].set(title="Position error", xlabel="Error (mm)", ylabel="Frequency")
    axes[1].hist(orientation_errors, bins=50, color="tab:red")
    axes[1].set(title="Orientation error", xlabel="Error (rad)", ylabel="Frequency")
    figure.suptitle(title)
    return figure


def plot_joint_errors(joint_angles, errors, ylabel, title):
    figure, axes = plt.subplots(3, 2, figsize=(16, 12))
    for i, axis in enumerate(axes.flat):
        axis.scatter(joint_angles[:, i], errors, s=10)
        axis.set(
            xlabel=f"Joint {i + 1} angle (rad)",
            ylabel=ylabel,
            title=f"Joint {i + 1}",
        )
    figure.suptitle(title)
    return figure


def compare_FK(num_samples=NUM_SAMPLES, rng_seed=RNG_SEED):
    """Sweep one fixed set of configurations through vendor, PoX, and DH FK."""
    arm = Lite6Arm()
    if arm.xarm is None:
        raise RuntimeError("The xArm SDK is unavailable; cannot compare against the vendor FK.")

    rng = np.random.default_rng(rng_seed)
    joint_angles = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(num_samples, 6))

    pox_vendor_position_errors = []
    pox_vendor_orientation_errors = []
    pox_dh_position_errors = []
    pox_dh_orientation_errors = []

    for q in joint_angles:
        code, pose_vendor = arm.xarm.get_forward_kinematics(q)
        if code != 0:
            raise RuntimeError(f"Vendor forward kinematics failed for q={q}: error code {code}")

        T_pox = FK_pox(q, M, S_list)
        T_dh = FK_dh(DH_STD, q, 6)

        pose_vendor = np.asarray(pose_vendor, dtype=float)
        R_vendor = R.from_euler("xyz", pose_vendor[3:], degrees=False).as_matrix()

        pox_vendor_position_errors.append(np.linalg.norm(T_pox[:3, 3] - pose_vendor[:3]))
        pox_vendor_orientation_errors.append(rotation_error(T_pox[:3, :3], R_vendor))
        pox_dh_position_errors.append(np.linalg.norm(T_pox[:3, 3] - T_dh[:3, 3]))
        pox_dh_orientation_errors.append(rotation_error(T_pox[:3, :3], T_dh[:3, :3]))

    pox_vendor_position_errors = np.asarray(pox_vendor_position_errors)
    pox_vendor_orientation_errors = np.asarray(pox_vendor_orientation_errors)
    pox_dh_position_errors = np.asarray(pox_dh_position_errors)
    pox_dh_orientation_errors = np.asarray(pox_dh_orientation_errors)

    print(f"Sample count: {num_samples}; random seed: {rng_seed}")
    print_error_statistics(
        "PoX compared with vendor FK",
        pox_vendor_position_errors,
        pox_vendor_orientation_errors,
    )
    print_error_statistics(
        "PoX-DH agreement on the same configurations",
        pox_dh_position_errors,
        pox_dh_orientation_errors,
    )

    figures = (
        (
            plot_histograms(
                "PoX compared with vendor FK",
                pox_vendor_position_errors,
                pox_vendor_orientation_errors,
            ),
            "pox_vendor_error_histograms.png",
        ),
        (
            plot_joint_errors(
                joint_angles,
                pox_vendor_orientation_errors,
                "Orientation error (rad)",
                "PoX-vendor orientation error by joint",
            ),
            "pox_vendor_orientation_errors_by_joint.png",
        ),
        (
            plot_joint_errors(
                joint_angles,
                pox_vendor_position_errors,
                "Position error (mm)",
                "PoX-vendor position error by joint",
            ),
            "pox_vendor_position_errors_by_joint.png",
        ),
        (
            plot_histograms(
                "PoX-DH agreement on identical configurations",
                pox_dh_position_errors,
                pox_dh_orientation_errors,
            ),
            "pox_dh_agreement_histograms.png",
        ),
    )
    for figure, filename in figures:
        save_figure(figure, filename)

    plt.show()


if __name__ == "__main__":
    compare_FK()

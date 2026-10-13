"""
Block detections: the one interface between perception and competition code.

In the sim, SimCamera.detect_blocks() (camera_sim.py) builds these from the
MuJoCo ground truth. On the real arm, Camera.detect_blocks() in camera.py,
which you write for Checkpoint 3, has to return the same list, so the code in
competition.py runs unchanged on both.

Units and frame: mm and rad in the world frame, which is the arm base frame
(origin at the centre of the arm mount, board surface at z = 0).
"""
from dataclasses import dataclass

import numpy as np

LARGE_MM = 38.0   # cube edge lengths
SMALL_MM = 25.0


def wrap_yaw(yaw):
    """Wrap a cube's yaw to [-pi/4, pi/4): a cube looks the same every 90 deg."""
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


@dataclass
class Block:
    color: str      # red, orange, yellow, green, blue or violet
    size: float     # edge length (mm), LARGE_MM or SMALL_MM
    x: float        # centre of the block (mm)
    y: float
    z: float
    yaw: float      # rotation about world z (rad), wrapped to [-pi/4, pi/4)

    @property
    def is_large(self):
        return self.size > (LARGE_MM + SMALL_MM) / 2

    def __str__(self):
        size = "large" if self.is_large else "small"
        return (f"{self.color} {size} at ({self.x:.0f}, {self.y:.0f}, {self.z:.0f}) mm, "
                f"yaw {np.degrees(self.yaw):.0f} deg")

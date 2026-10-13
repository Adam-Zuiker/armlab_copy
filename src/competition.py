"""
Competition events. The same code runs in the sim and on the real arm.

Run an event from a custom button along the bottom of the control station: the
state machine runs it in its own thread. To add an event, write it here and list
it in CUSTOM_BUTTONS in control_station.py.

An event is a function event(arm, camera):
  arm     Lite6Arm or SimArm, same API:
            arm.get_ik(pose), arm.get_ik_numerical(pose)  -> 6 joint angles (rad) or None
            arm.set_joint_angles(q, wait=True)            -> True when the move is done
            arm.open_gripper(wait=True), arm.close_gripper(wait=True)
                                                          -> nothing, run after earlier queued moves
            arm.get_joint_angles(), arm.get_ee_pose()     -> rad, and the flange pose
  camera  camera.detect_blocks() -> list[Block] (blocks.py)
            sim:  exact poses of the blocks a camera looking straight down sees
            real: your Camera.detect_blocks() in camera.py must return the same list

Poses are [x, y, z, roll, pitch, yaw] in mm and rad, in the world frame = arm base
frame (z = 0 on the board). get_ik() and get_ee_pose() use the flange pose. A
top-down grasp is roll = pi, pitch = 0, with the flange GRASP_OFFSET_MM above the
grasp point.

After STOP, set_joint_angles returns False, and so do move_to, pick and place
below. Check what they return and give up early.
"""
import random

import numpy as np

from blocks import wrap_yaw
from kinematics import Q_DEFAULT

GRASP_OFFSET_MM = 85.0    # flange to the middle of the finger pads: the grasp point
FINGER_TIP_MM = 96.4      # flange to the finger tips
TIP_CLEARANCE_MM = 3.0    # keep the finger tips this far above the surface under a block
RELEASE_MM = 2.0          # let go of a block this far above where it lands
HOVER_MM = 60.0           # approach and retreat height above a grasp or a place


def top_down_pose(x, y, z_grasp, yaw):
    """Flange pose putting the grasp point at (x, y, z_grasp), jaws pointing down.

    The jaws close along the flange y axis, which for this pose is the world
    direction (sin yaw, -cos yaw).
    """
    return [x, y, z_grasp + GRASP_OFFSET_MM, np.pi, 0.0, yaw]


def jaw_yaw(x, y, block_yaw):
    """Gripper yaw that squares the jaws with a block of yaw block_yaw at (x, y).

    Any yaw a multiple of 90 deg away grips a cube just as well. Take the one
    closest to the direction of (x, y) from the base, which keeps joint 6 near zero.
    """
    radial = np.arctan2(y, x)
    return radial + wrap_yaw(block_yaw - radial)


def move_to(arm, pose):
    """Blocking joint move to a flange pose, with your IK_geometric, or IK_numerical
    if that finds nothing. False if unreachable, failed or stopped."""
    q = arm.get_ik(pose)
    if q is None:
        q = arm.get_ik_numerical(pose)
    if q is None:
        print(f"No IK solution for {np.round(pose, 1)}")
        return False
    return arm.set_joint_angles(q, wait=True)


def grasp_height(block):
    """Grasp point height for a block: its centre, or higher if that keeps the finger
    tips clear of the surface the block stands on (small blocks)."""
    bottom = block.z - block.size / 2
    return max(block.z, bottom + TIP_CLEARANCE_MM + FINGER_TIP_MM - GRASP_OFFSET_MM)


def pick(arm, block, hover_height=None):
    """Grasp a block from above and lift it. hover_height raises the grasp point to
    that height afterwards, to carry the block over others. True on success."""
    z = grasp_height(block)
    yaw = jaw_yaw(block.x, block.y, block.yaw)
    arm.open_gripper()  # queued: opens on the way to the hover pose
    if not move_to(arm, top_down_pose(block.x, block.y, z + HOVER_MM, yaw)):
        return False
    if not move_to(arm, top_down_pose(block.x, block.y, z, yaw)):
        return False
    arm.close_gripper(wait=True)
    lift_z = z + HOVER_MM if hover_height is None else max(z + HOVER_MM, hover_height)
    return move_to(arm, top_down_pose(block.x, block.y, lift_z, yaw))


def place(arm, block, x, y, z_surface, yaw, hover_height=None):
    """Set the held block down centred at (x, y) on a surface at height z_surface
    (0 for the board, the top of another block to stack on it), then back off
    upward. The block lands square to the jaws, which point along yaw. hover_height
    is the grasp point height to arrive at, as in pick(). True on success."""
    # Hold the block as pick() did: the grasp point grasp_height(block) - block.z above its centre.
    z = z_surface + block.size / 2 + (grasp_height(block) - block.z) + RELEASE_MM
    arrive_z = z + HOVER_MM if hover_height is None else max(z + HOVER_MM, hover_height)
    if not move_to(arm, top_down_pose(x, y, arrive_z, yaw)):
        return False
    if not move_to(arm, top_down_pose(x, y, z, yaw)):
        return False
    arm.open_gripper(wait=True)
    return move_to(arm, top_down_pose(x, y, z + HOVER_MM, yaw))


# ----------------------------------------------------------------------
# Event 0 (sim warm-up)
# ----------------------------------------------------------------------

# Each click moves one block, a random one in front of the arm, onto the lowest slot
# just behind it (the first one on a tie). The slots fill in order, then a second
# layer goes on top. Slots with |y| < 200 mm would bring the gripper close to the arm
# base, and |y| > 400 mm is out of reach.
EVENT0_PLACE_X_MM = -25.0
EVENT0_PLACE_Y_MM = (-400.0, -300.0, -200.0, 200.0, 300.0, 400.0)   # 100 mm apart
EVENT0_HOVER_HEIGHT_MM = 155.0   # grasp point height for carrying, well above the blocks


def event_0(arm, camera):
    """Event 0: pick a random block in front of the arm (+x half-plane), place it on
    top of the lowest slot behind the arm, square to the board, and go back to the
    initial pose."""
    seen = camera.detect_blocks()
    blocks = [b for b in seen if b.x > 0]
    if not blocks:
        print("No blocks left in front of the arm")
        return
    x = EVENT0_PLACE_X_MM
    # Height of each slot: the top of the block standing there, 0 for the bare board.
    tops = [max((b.z + b.size / 2 for b in seen if abs(b.x - x) < 20 and abs(b.y - y) < 20), default=0.0)
            for y in EVENT0_PLACE_Y_MM]
    slot = tops.index(min(tops))
    y = EVENT0_PLACE_Y_MM[slot]
    block = random.choice(blocks)
    print(f"Picking the {block}")
    if not pick(arm, block, hover_height=EVENT0_HOVER_HEIGHT_MM):
        print("Pick failed")
        return
    if not place(arm, block, x, y, tops[slot], jaw_yaw(x, y, block_yaw=0.0), hover_height=EVENT0_HOVER_HEIGHT_MM):
        print("Place failed")
        return
    print(f"Placed at ({x:.0f}, {y:.0f}) mm")

    # Return to home pose
    arm.set_joint_angles(Q_DEFAULT, wait=True)

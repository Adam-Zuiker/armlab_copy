"""
Event 2 Level 3: line up the cubes in rainbow order - large cubes in one line behind the
arm at y < 0, small cubes in another at y > 0 - and leave the distractors (other shapes)
where they are. Stacks are up to three high.

Every color has its own slot in its line, so the blocks can go in any order: the highest
visible cube is picked first, which takes the stacks apart from the top.

One pick & place per call to step(), so the state machine can update the status between blocks.
After each place the arm stays above the lines and senses again from there.
"""
import time

import cv2
import numpy as np

from blocks import STACK_TOL_MM, stack_error
from kinematics import Q_DEFAULT

BLOCK_H = {"big": 38.0, "small": 25.0}

ROYGBV = ["red", "orange", "yellow", "green", "blue", "purple"]
RANK = {color: i for i, color in enumerate(ROYGBV)}

# Lines (mm): along y at x = LINE_X, red nearest the base. Large line 243 mm long,
# small line 180 mm, both under the 300 mm limit.
LINE_X = -50.0
LINE_Y0 = 195.0                           # |y| of the red slot
PITCH = {"big": 43.0, "small": 31.0}      # 5 mm / 6 mm gaps
LINE_SIDE = {"big": -1.0, "small": 1.0}   # sign of y for each line

# Carry blocks with the grasp point this high (mm): clears a held large block over a
# three-high stack of large blocks with 25 mm to spare.
HOVER_Z = 3 * BLOCK_H["big"] + BLOCK_H["big"] / 2 + 25.0

# What makes a detection a cube rather than a distractor.
SIDE_TOL_MM = 7.0      # top face no smaller than a cube's edge length less this,
MERGE_TOL_MM = 12.0    # and no bigger than it plus this: an off-center cube underneath can merge into the outline
MIN_FILL = 0.85        # contour area / its min-area rectangle: square ~1, circle 0.79, triangle 0.5

# Jaw clearance when picking: a cube can be gripped from two directions 90 deg apart,
# take the one that keeps the fingers farthest from the neighbors.
FINGER_MM = 12.0       # finger center beyond the cube face with the jaws open
NEIGHBOR_MM = 100.0    # detections farther than this from the cube don't matter

SETTLE_S = 0.5      # after the arm leaves the camera's view, let detections refresh
RESENSE_S = 0.2     # same, sensing from above the lines after a place
MAX_TRIES = 3       # picks of one block before giving up on it


def slot(size, color):
    """(x, y) of a cube's place in its line."""
    return LINE_X, LINE_SIDE[size] * (LINE_Y0 + PITCH[size] * RANK[color])


def is_cube(d):
    """True for a detection that looks like one of the 12 cubes, False for a distractor."""
    if d["color"] not in RANK:
        return False
    side = BLOCK_H[d["size"]]
    if not -SIDE_TOL_MM <= d.get("side_mm", side) - side <= MERGE_TOL_MM:
        return False
    if stack_error(d["world"][2] - side) >= STACK_TOL_MM:   # must stand on the board or on cubes
        return False
    if "contour" in d:                           # the sim reports cubes only, without contours
        _, (w, h), _ = cv2.minAreaRect(d["contour"])
        if cv2.contourArea(d["contour"]) < MIN_FILL * w * h:
            return False
    return True


class Event2Level3:

    def __init__(self, sm):
        self.sm = sm
        self.reset()

    def reset(self):
        self.filled = set()  # (color, size) of the cubes in their slots
        self.skipped = set() # (color, size) given up on
        self.at_home = False
        self.last = None     # (color, size) of the block placed last, until its place is confirmed
        self.tries = {}      # (color, size) -> picks so far

    def _sense(self):
        """Everything still in front of the arm (+x half-plane): cubes and distractors."""
        return [d for d in self.sm.camera.block_detections if d["world"][0] > 0]

    def _todo(self, seen):
        """Cubes in seen that still need a place in a line."""
        return [d for d in seen if is_cube(d) and (d["color"], d["size"]) not in self.filled | self.skipped]

    def _go_home_and_sense(self):
        self.sm._move(list(Q_DEFAULT))               # out of the camera's view
        self.at_home = True
        time.sleep(SETTLE_S)
        return self._sense()

    def _jaw_yaw(self, d, seen):
        """Yaw for sm._pick that squares the jaws with cube d, of its two choices the one
        whose fingers come down farthest from the other detections."""
        x, y = d["world"][:2]
        reach = BLOCK_H[d["size"]] / 2 + FINGER_MM
        others = [o for o in seen if 1.0 < np.hypot(o["world"][0] - x, o["world"][1] - y) < NEIGHBOR_MM]

        def clearance(yaw):
            # sm._pick turns the flange to -yaw, and the jaws close along (sin, -cos) of that
            jaw = np.array([np.sin(-yaw), -np.cos(-yaw)])
            fingers = [np.array([x, y]) + s * reach * jaw for s in (-1, 1)]
            return min((np.hypot(*(f - o["world"][:2])) - o.get("side_mm", BLOCK_H["big"]) / 2
                        for f in fingers for o in others), default=np.inf)

        yaw = self.sm._block_yaw(d)
        alt = yaw + np.pi / 2 if yaw < 0 else yaw - np.pi / 2
        return alt if clearance(alt) > clearance(yaw) else yaw

    def _summary(self):
        msg = f"{len(self.filled)} of 12 cubes lined up"
        if self.skipped:
            msg += ", skipped " + ", ".join(f"{c} {s}" for c, s in sorted(self.skipped))
        return msg + "."

    def step(self):
        """Line up one cube. Returns a message when the event is finished, otherwise None."""
        sm = self.sm
        if self.at_home or self.last is None:
            seen = self._go_home_and_sense()         # first block: start from a clear view
        else:
            time.sleep(RESENSE_S)                    # sense from above the lines
            seen = self._sense()

        # Confirm the last place: no gripper feedback, so a missed pick shows up as
        # the block still waiting in front of the arm. Cubes only: a distractor may share its color.
        todo = self._todo(seen)
        if self.last is not None:
            if not any((d["color"], d["size"]) == self.last for d in todo):
                self.filled.add(self.last)
                todo = self._todo(seen)
            self.last = None

        if not todo and not self.at_home:
            seen = self._go_home_and_sense()         # make sure the arm was not hiding a block
            todo = self._todo(seen)
        if not todo:
            sm._move(list(Q_DEFAULT))
            return "Event 2 L3 done - " + self._summary()

        d = max(todo, key=lambda d: d["world"][2])   # highest first: stacks come apart from the top
        key = (d["color"], d["size"])
        self.tries[key] = self.tries.get(key, 0) + 1
        if self.tries[key] > MAX_TRIES:
            self.skipped.add(key)
            sm.status_message = f"Event 2 L3: could not pick the {key[0]} {key[1]} block, skipping it..."
            return None

        self.at_home = False
        sm._pick(d["world"], yaw=self._jaw_yaw(d, seen), hover_z=HOVER_Z)
        if not sm.holding_block:
            self.skipped.add(key)
            sm.status_message = f"Event 2 L3: cannot reach the {key[0]} {key[1]} block, skipping it..."
            return None

        x, y = slot(d["size"], d["color"])
        sm.held_yaw = -LINE_SIDE[d["size"]] * np.pi / 2   # jaws close across the line, clear of the neighbors
        sm._place((x, y, 0.0), go_home=False, hover_z=HOVER_Z, snap=False)
        if sm.holding_block:
            return "Event 2 L3 stopped - could not reach a slot, still holding the block. " + self._summary()

        self.last = key
        sm.status_message = f"Event 2 L3: placed the {key[0]} {key[1]} block..."
        return None

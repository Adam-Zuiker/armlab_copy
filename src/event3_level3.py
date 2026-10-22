"""
Event 2 Level 3: line up the cubes in rainbow order - large cubes in one line behind the
arm at y < 0, small cubes in another at y > 0 - and leave the distractors (other shapes)
where they are. Stacks are up to three high.

Every color has its own slot in its line, so the blocks can go in any order: the highest
visible cube is picked first, which takes the stacks apart from the top. A slot takes one
block only: a cube whose color's slot is taken (a color misread) goes to the nearest free one.

One pick & place per call to step(), so the state machine can update the status between blocks.
After each place the arm stays above the lines and senses again from there.
"""
import time

import cv2
import numpy as np

from kinematics import Q_DEFAULT

BLOCK_H = {"big": 38.0, "small": 25.0}

ROYGBV = ["red", "orange", "yellow", "green", "blue", "purple"]
RANK = {color: i for i, color in enumerate(ROYGBV)}

# Lines (mm): along y at x = LINE_X, red nearest the base. Large line 283 mm long,
# small line 215 mm, both under the 300 mm limit. The far large slot, |y| = 425, is 10 mm
# inside the arm's reach there.
LINE_X = -50.0
LINE_Y0 = 180.0                           # |y| of the red slot
PITCH = {"big": 49.0, "small": 38.0}      # 11 mm / 13 mm gaps
LINE_SIDE = {"big": -1.0, "small": 1.0}   # sign of y for each line

# Carry blocks with the grasp point this high (mm): clears a held large block over a
# three-high stack of large blocks with 25 mm to spare.
HOVER_Z = 3 * BLOCK_H["big"] + BLOCK_H["big"] / 2 + 25.0

# What makes a detection a cube rather than a distractor.
SIDE_TOL_MM = 7.0      # top face no smaller than a cube's edge length less this,
MERGE_TOL_MM = 10.0    # and no bigger than it plus this: an off-center cube underneath can merge into the outline
MIN_FILL = 0.85        # contour area / its min-area rectangle: square ~1, circle 0.79, triangle 0.5
HEIGHT_TOL_MM = 15.0   # it stands this close to the board or the top of a two-cube stack, or between

# Jaw clearance when picking: a cube can be gripped from two directions 90 deg apart,
# take the one that keeps the fingers farthest from the neighbors.
FINGER_MM = 12.0       # finger center beyond the cube face with the jaws open
NEIGHBOR_MM = 100.0    # detections farther than this from the cube don't matter

SETTLE_S = 0.5      # after the arm leaves the camera's view, let detections refresh
RESENSE_S = 0.2     # same, sensing from above the lines after a place
MAX_TRIES = 3       # picks of one block before giving up on it
SAME_SPOT_MM = 15.0 # a detection this close to a pick (in x, y and in height) is the block picked there


def slot(size, i):
    """(x, y) of slot i of a line: slot RANK[color] is that color's place."""
    return LINE_X, LINE_SIDE[size] * (LINE_Y0 + PITCH[size] * i)


def reject_reason(d):
    """Why detection d is not one of the 12 cubes (a distractor), or None if it is one."""
    if d["color"] not in RANK:
        return f"color {d['color']}"
    side = BLOCK_H[d["size"]]
    side_mm = d.get("side_mm", side)
    if not -SIDE_TOL_MM <= side_mm - side <= MERGE_TOL_MM:
        return f"top face {side_mm:.0f} mm wide for a {d['size']} cube"
    below = d["world"][2] - side                 # top of whatever the block stands on
    if not -HEIGHT_TOL_MM < below < 2 * BLOCK_H["big"] + HEIGHT_TOL_MM:
        return f"top at {d['world'][2]:.0f} mm, too {'low' if below < 0 else 'high'} for a {d['size']} cube"
    if "contour" in d:                           # the sim reports cubes only, without contours
        _, (w, h), _ = cv2.minAreaRect(d["contour"])
        fill = cv2.contourArea(d["contour"]) / max(w * h, 1e-6)
        if fill < MIN_FILL:
            return f"outline {fill:.0%} of its rectangle, not square"
    return None


def is_cube(d):
    """True for a detection that looks like one of the 12 cubes, False for a distractor."""
    return reject_reason(d) is None


class Event2Level3:

    def __init__(self, sm):
        self.sm = sm
        self.reset()

    def reset(self):
        # Color of the block placed in each slot, None while free. A slot only ever takes one
        # block: one detected in the wrong color, whose slot is taken, goes to the nearest free one.
        self.slots = {"big": [None] * len(ROYGBV), "small": [None] * len(ROYGBV)}
        self.misses = []     # world (x, y, z) of picks that left the block where it was
        self.skipped = []    # (world (x, y, z), label) of blocks given up on
        self.at_home = False
        self.last = None     # (detection, size, slot) of the block placed last, until its place is confirmed

    def _sense(self):
        """Everything still in front of the arm (+x half-plane): cubes and distractors."""
        return [d for d in self.sm.camera.block_detections if d["world"][0] > 0]

    @staticmethod
    def _at(d, p):
        """True if detection d is at world point p: the same spot and the same height."""
        x, y, z = d["world"][:3]
        return np.hypot(x - p[0], y - p[1]) < SAME_SPOT_MM and abs(z - p[2]) < SAME_SPOT_MM

    def _todo(self, seen):
        """Cubes in seen that still need a place in a line."""
        return [d for d in seen if is_cube(d) and not any(self._at(d, p) for p, _ in self.skipped)]

    def _free_slot(self, size, color):
        """The color's slot in the size's line, or the free one nearest it, or None if the line is full."""
        free = [i for i, c in enumerate(self.slots[size]) if c is None]
        return min(free, key=lambda i: abs(i - RANK[color])) if free else None

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

    def _summary(self, seen=None):
        """Result so far. With seen (the last detections), also what was left in front and why."""
        placed = [(i, c) for line in self.slots.values() for i, c in enumerate(line) if c is not None]
        msg = f"{len(placed)} of 12 cubes lined up"
        out_of_order = sum(c != ROYGBV[i] for i, c in placed)
        if out_of_order:
            msg += f" ({out_of_order} out of order: their color's slot was taken)"
        if self.skipped:
            msg += ", skipped " + ", ".join(label for _, label in self.skipped)
        msg += "."
        if seen is None:
            return msg
        if not seen:
            return msg + " The camera sees no blocks in front of the arm - is it calibrated?"
        left = [(d, reject_reason(d)) for d in seen if not is_cube(d)]
        for d, why in left:
            x, y, z = d["world"][:3]
            print(f"Event 2 L3: left the {d['color']} {d['size']} detection at ({x:.0f}, {y:.0f}, {z:.0f}) - {why}")
        if left:
            msg += f" Left {len(left)} detection(s) as distractors - reasons printed in the terminal."
        return msg

    def step(self):
        """Line up one cube. Returns a message when the event is finished, otherwise None."""
        sm = self.sm
        if self.at_home or self.last is None:
            seen = self._go_home_and_sense()         # first block: start from a clear view
        else:
            time.sleep(RESENSE_S)                    # sense from above the lines
            seen = self._sense()

        # Confirm the last place: no gripper feedback, so a missed pick shows up as a block
        # still where it was picked, at the same height. Not by color: two blocks can be
        # detected in the same color, and the block under the picked one sits lower.
        todo = self._todo(seen)
        if self.last is not None:
            d, size, i = self.last
            if any(self._at(o, d["world"]) for o in todo):
                self.slots[size][i] = None
                self.misses.append(d["world"][:3])
            self.last = None

        if not todo and not self.at_home:
            seen = self._go_home_and_sense()         # make sure the arm was not hiding a block
            todo = self._todo(seen)
        if not todo:
            sm._move(list(Q_DEFAULT))
            return "Event 2 L3 done - " + self._summary(seen)

        d = max(todo, key=lambda d: d["world"][2])   # highest first: stacks come apart from the top
        label = f"{d['color']} {d['size']}"
        p = d["world"][:3]
        if sum(self._at(d, m) for m in self.misses) >= MAX_TRIES:
            self.skipped.append((p, label))
            sm.status_message = f"Event 2 L3: could not pick the {label} block, skipping it..."
            return None
        i = self._free_slot(d["size"], d["color"])
        if i is None:
            self.skipped.append((p, label))
            sm.status_message = f"Event 2 L3: the {d['size']} line is full, skipping the {label} block..."
            return None

        self.at_home = False
        sm._pick(p, yaw=self._jaw_yaw(d, seen), hover_z=HOVER_Z)
        if not sm.holding_block:
            self.skipped.append((p, label))
            sm.status_message = f"Event 2 L3: cannot reach the {label} block, skipping it..."
            return None

        x, y = slot(d["size"], i)
        sm.held_yaw = -LINE_SIDE[d["size"]] * np.pi / 2   # jaws close across the line, clear of the neighbors
        sm._place((x, y, 0.0), go_home=False, hover_z=HOVER_Z, snap=False)
        if sm.holding_block:
            return "Event 2 L3 stopped - could not reach a slot, still holding the block. " + self._summary()

        self.slots[d["size"]][i] = d["color"]
        self.last = (d, d["size"], i)
        sm.status_message = f"Event 2 L3: placed the {label} block..."
        return None

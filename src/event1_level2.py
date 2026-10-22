"""
Event 1 Level 2: stack blocks by size in rainbow order - large blocks on LARGE_SLOT,
small blocks on SMALL_SLOT, large ones first.

One pick & place per call to step(), so the state machine can update the status between blocks.
After each place the arm stays above the stacks and senses again from there, rather
than going home.
"""
import time

import numpy as np

from kinematics import Q_DEFAULT

BLOCK_H = {"big": 38.0, "small": 25.0}

# Stack positions (x, y) in mm.
LARGE_SLOT = (-50.0, -250.0)
SMALL_SLOT = (-50.0, 250.0)

ROYGBV = ["red", "orange", "yellow", "green", "blue", "purple"]
RANK = {color: i for i, color in enumerate(ROYGBV)}

SETTLE_S = 0.5      # after the arm leaves the camera's view, let detections refresh
RESENSE_S = 0.2     # same, sensing from above the stacks after a place
MAX_TRIES = 3       # picks of one block before giving up on it


class Event1Level2:

    def __init__(self, sm):
        self.sm = sm
        self.reset()

    def reset(self):
        self.stack_h = {"big": 0.0, "small": 0.0}   # top of each slot's stack (mm)
        self.at_home = False
        self.last = None     # (color, size) of the block placed last, until its place is confirmed
        self.tries = {}      # (color, size) -> picks so far

    def _sense(self):
        """Blocks still to sort, i.e. in front of the arm (+x half-plane)."""
        return [d for d in self.sm.camera.block_detections if d["world"][0] > 0]

    def _go_home_and_sense(self):
        self.sm._move(list(Q_DEFAULT))               # out of the camera's view
        self.at_home = True
        time.sleep(SETTLE_S)
        return self._sense()

    def step(self):
        """Sort one block. Returns a message when the event is finished, otherwise None."""
        sm = self.sm
        if self.at_home or self.last is None:
            todo = self._go_home_and_sense()         # first block: start from a clear view
        else:
            time.sleep(RESENSE_S)                    # sense from above the stacks
            todo = self._sense()

        # Confirm the last place: no gripper feedback, so a missed pick shows up as
        # the block still waiting in front of the arm.
        if self.last is not None:
            color, size = self.last
            if not any(d["color"] == color and d["size"] == size for d in todo):
                self.stack_h[size] += BLOCK_H[size]
            self.last = None

        if not todo and not self.at_home:
            todo = self._go_home_and_sense()         # make sure the arm was not hiding a block
        if not todo:
            return f"Event 1 L2 done - large stack {self.stack_h['big']:.0f} mm, small {self.stack_h['small']:.0f} mm."

        big = [d for d in todo if d["size"] == "big"]
        candidates = big if big else todo
        d = min(candidates, key=lambda d: RANK.get(d["color"], len(ROYGBV)))
        slot = LARGE_SLOT if d["size"] == "big" else SMALL_SLOT

        key = (d["color"], d["size"])
        self.tries[key] = self.tries.get(key, 0) + 1
        if self.tries[key] > MAX_TRIES:
            return f"Event 1 L2 stopped - could not pick the {d['color']} {d['size']} block in {MAX_TRIES} tries."

        self.at_home = False
        sm._pick(d["world"])
        if not sm.holding_block:
            return "Event 1 L2 stopped - could not reach a block."

        sm.held_yaw = 0.0                            # square the block up in its slot
        sm._place((slot[0], slot[1], self.stack_h[d["size"]]), go_home=False)
        if sm.holding_block:
            return "Event 1 L2 stopped - could not reach a slot, still holding the block."

        self.last = key
        sm.status_message = f"Event 1 L2: placed the {d['color']} {d['size']} block..."
        return None

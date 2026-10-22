"""
Event 1: sort blocks by size - large blocks to y > 0, small blocks to y < 0.

One pick & place per call to step(), so the state machine can update the status between blocks.
"""
import time

import numpy as np

from kinematics import Q_DEFAULT

# Drop slots (x, y) in mm, filled in list order.
LARGE_SLOTS = [(x, y) for y in (-200.0, -300.0, -400.0) for x in (-50.0, )]
SMALL_SLOTS = [(x, -y) for x, y in LARGE_SLOTS]

SORTED_MM = 25.0    # a block this close to a filled slot is already sorted
SETTLE_S = 0.5      # after the arm leaves the camera's view, let detections refresh


class Event1:

    def __init__(self, sm):
        self.sm = sm
        self.reset()

    def reset(self):
        self.free = {"big": list(LARGE_SLOTS), "small": list(SMALL_SLOTS)}
        self.filled = []

    def _is_sorted(self, d):
        x, y = d["world"][:2]
        return any(np.hypot(x - sx, y - sy) < SORTED_MM for sx, sy in self.filled)

    def step(self):
        """Sort one block. Returns a message when the event is finished, otherwise None."""
        sm = self.sm
        sm._move(list(Q_DEFAULT))                    # out of the camera's view
        time.sleep(SETTLE_S)

        todo = [d for d in sm.camera.block_detections if not self._is_sorted(d)]
        if not todo:
            return f"Event 1 done - sorted {len(self.filled)} block(s)."

        d = max(todo, key=lambda d: d["world"][2])   # highest first, so stacks come apart from the top
        if not self.free[d["size"]]:
            return f"Event 1 stopped - no free slot for a {d['size']} block."
        slot = self.free[d["size"]].pop(0)

        sm._pick(d["world"])
        if not sm.holding_block:
            return "Event 1 stopped - could not reach a block."

        sm.held_yaw = 0.0                            # square the block up in its slot
        sm._place((slot[0], slot[1], 0.0))
        if sm.holding_block:
            return "Event 1 stopped - could not reach a slot, still holding the block."

        self.filled.append(slot)
        sm.status_message = f"Event 1: sorted {len(self.filled)} block(s)..."
        return None

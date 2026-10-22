"""
Event 2: sort blocks by size - large blocks to y > 0, small blocks to y < 0.

One pick & place per call to step(), so the state machine can update the status between blocks.
"""
import time

import numpy as np

from kinematics import Q_DEFAULT

BLOCK_H = {"big": 38.0, "small": 25.0}

# Drop slots (x, y) in mm, filled in list order.
LARGE_SLOT = (-50.0, -200.0)
SMALL_SLOT = (-50.0, -300.0)
BIG_HEIGHT = 2
SMALL_HEIGHT = 2

SORTED_MM = 25.0    # a block this close to a filled slot is already sorted
SETTLE_S = 0.5      # after the arm leaves the camera's view, let detections refresh


class Event1Level2:

    def __init__(self, sm):
        self.sm = sm
        self.reset()

    def reset(self):
        return

    # def _is_sorted(self, d):
    #     x, y = d["world"][:2]
    #     return any(np.hypot(x - sx, y - sy) < SORTED_MM for sx, sy in self.filled)

    def step(self):
        """Sort one block. Returns a message when the event is finished, otherwise None."""
        sm = self.sm
        sm._move(list(Q_DEFAULT))                    # out of the camera's view
        time.sleep(SETTLE_S)

        # detections.append({
        #     "contour": contour,
        #     "center_td": (cx, cy),
        #     "side_mm": side,
        #     "size": size,
        #     "angle_degree": angle_deg,
        #     "color": color,
        #     "world": p_world
        # })

        #print(f" block detections: {[d['world'][0] for d in sm.camera.block_detections]}")

        todo = [d for d in sm.camera.block_detections if d["world"][0] > 0]
        if not todo:
            #return f"Event 2 done - sorted {len(self.filled)} block(s)."
            print(todo)
            return("iykyk ;)")

        ROYGBV = ["red", "orange", "yellow", "green", "blue", "purple"]
        rank = {color: i for i, color in enumerate(ROYGBV)}

        big = [d for d in todo if d["size"] == "big"]

        if len(big) == 0:
            small = [d for d in todo if d["size"] == "small"]
            color_sorted = sorted(small, key=lambda x: rank[x["color"]])
            slot = SMALL_SLOT
        else:
            color_sorted = sorted(big, key=lambda x: rank[x["color"]])
            slot = LARGE_SLOT


        d = color_sorted[0]
        # if not self.free[d["size"]]:
        #     return f"Event 2 stopped - no free slot for a {d['size']} block."
        # slot = self.free[d["size"]].pop(0)

        sm._pick(d["world"])
        if not sm.holding_block:
            return "Event 2 stopped - could not reach a block."

        sm.held_yaw = 0.0                            # square the block up in its slot
        sm._place((slot[0], slot[1], 0.0))
        if sm.holding_block:
            return "Event 2 stopped - could not reach a slot, still holding the block."

        if slot == SMALL_SLOT:
            SMALL_HEIGHT += BLOCK_H["small"]
        elif slot == BIG_HEIGHT:
            SMALL_HEIGHT += BLOCK_H["big"]

        # self.filled.append(slot)
        #sm.status_message = f"Event 2: sorted {len(self.filled)} block(s)..."
        return None

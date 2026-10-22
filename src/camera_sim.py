"""
Camera stand-in for --sim, maintained by the course staff.

SimCamera is the regular Camera (camera.py) with the RealSense left out, so the
GUI behaves as with no camera plugged in. detect_blocks() returns what the sim
reports a camera looking straight down would see: the exact pose of every block
on the board with no other block above its centre. Competition code calls
camera.detect_blocks() the same way in sim and on the real arm.

    * GET  : ufactory_lite6_world (sim_world_t, ~50 Hz)
    * SEND : ufactory_lite6_reset (sim_reset_t), for Reset Board (Sim)
"""
import os
import sys
import threading
import time

# Temp fix: add mujoco_sim to path
_MUJOCO_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_sim")
if _MUJOCO_SIM_DIR not in sys.path:
    sys.path.insert(0, _MUJOCO_SIM_DIR)

import lcm
import numpy as np
from lite6_sim.lcm_msgs import sim_reset_t, sim_world_t

from blocks import Block
from camera import Camera

WORLD_CHANNEL = "ufactory_lite6_world"
RESET_CHANNEL = "ufactory_lite6_reset"
LCM_URL = "udpm://239.255.76.67:7667?ttl=0"  # same as lite6arm_sim
SETTLE_S = 0.5   # let the blocks settle after a reset


class SimCamera(Camera):

    def __init__(self):
        super().__init__()
        if self.camera_connected:
            # Never mix a real L515 into the sim.
            self.stop()
        self.camera_connected = False
        self._world = None   # latest sim_world_t
        self._lc = lcm.LCM(LCM_URL)
        self._lc.subscribe(WORLD_CHANNEL, self._on_world)
        threading.Thread(target=self._lcm_loop, daemon=True).start()

    def _lcm_loop(self):
        while True:
            self._lc.handle_timeout(100)

    def _on_world(self, channel, data):
        try:
            self._world = sim_world_t.decode(data)
        except ValueError:
            pass  # a sim built with other LCM types

    def detect_blocks(self):
        """list[Block] (blocks.py): exact poses of the blocks visible from above."""
        msg = self._world
        if msg is None:
            return []
        return [Block(color, size * 1000.0, pos[0] * 1000.0, pos[1] * 1000.0, pos[2] * 1000.0, yaw)
                for color, size, pos, yaw in zip(msg.color, msg.size, msg.pos, msg.yaw)]

    @property
    def block_detections(self):
        """detect_blocks() in the dict format of Camera.find_blocks(), for the state machine's events."""
        return [{
            "size": "big" if b.is_large else "small",
            "side_mm": b.size,
            "angle_degree": (-np.degrees(b.yaw)) % 90.0,
            "color": "purple" if b.color == "violet" else b.color,
            "world": np.array([b.x, b.y, b.z + b.size / 2]),   # top-face centre, like the real camera
        } for b in self.detect_blocks()]

    @block_detections.setter
    def block_detections(self, _):
        pass   # Camera.__init__ / find_blocks() assign it; the sim's blocks come from LCM instead

    def reset_board(self, keyframe):
        """Sim only: put every block where keyframe (armlab_scene.xml) places it."""
        msg = sim_reset_t()
        msg.timestamp = time.time_ns()
        msg.keyframe = keyframe
        self._lc.publish(RESET_CHANNEL, msg.encode())
        time.sleep(SETTLE_S)


def reset_board_event0(arm, camera):
    """Custom button (sim only): arm to the initial pose, clear of the blocks, then the
    Event 0 layout."""
    if not isinstance(camera, SimCamera):
        raise RuntimeError("works in the sim only")
    if not arm.initialize():
        raise RuntimeError("the arm did not initialize")
    camera.reset_board("event0")

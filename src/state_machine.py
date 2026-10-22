"""
State machine for the arm lab runtime shell.
"""
import time

from PyQt5.QtCore import QThread, pyqtSignal
# import lite6arm
import numpy as np

from kinematics import Q_DEFAULT
from camera_sim import reset_board_event0
from event1 import Event1
from event1_level2 import Event1Level2
from event3_level3 import Event2Level3

# --- pick & place geometry (mm) ---
FLANGE_TO_GRASP_MM = 85.0      # grasp point is 85 mm below the flange along the tool axis
FLANGE_TO_TIP_MM = 96.4        # finger tips are 96.4 mm below the flange
GRASP_RAISE_MM = 5.0           # grip this far above the block's center (tune: higher = grip nearer the top)
TIP_CLEARANCE_MM = 3.0         # and keep the finger tips at least this far above what the block stands on
APPROACH_CLEARANCE_MM = 100.0  # flange stays this far above the grasp/place pose when moving sideways
PLACE_DROP_MM = 5.0            # release the block slightly above the surface, not pressed into it
PLACE_SETTLE_S = 0.3           # let the arm stop swaying before letting go of a block
BLOCK_H = {"big": 38.0, "small": 25.0}
SNAP_RADIUS_MM = 40.0          # a click within this distance of a detected block snaps to that block

class StateMachine:

    def __init__(self, arm, camera):
        self.arm = arm
        self.camera = camera
        self.status_message = "Idle - waiting for input."
        self.current_state = "idle"
        self.next_state = "idle"

        self._handlers = {
            "initial_pose":      self.initial_pose,
            "sleep_arm":         self.sleep_arm,
            "estop":             self.estop,
            "calibrate":         self.calibrate,
            "add_waypoint":      self.add_waypoint,
            "clear_waypoints":   self.clear_waypoints,
            "playback_waypoints":self.playback_waypoints,
            "direct_control":    self.direct_control,
            "pick_place":        self.pick_place,
            "check_3d":          self.check_3d,
            "event1":            self.event1,
            "event1l2":          self.event1l2,
            "event2l3":          self.event2l3,
            "reset_board":       self.reset_board,
        }

        self.waypoints = []

        # pick & place
        self.holding_block = False
        self.held_block_h = BLOCK_H["big"]
        self.held_grasp_above_bottom = BLOCK_H["big"] / 2.0
        self.held_yaw = 0.0

        self.event1_task = Event1(self)
        self.event1l2_task = Event1Level2(self)
        self.event2l3_task = Event2Level3(self)

    def set_next_state(self, state):
        self.next_state = state

    def _go_idle(self, message):
        self.status_message = message
        self.current_state = "idle"
        self.next_state = "idle"

    def run(self):
        self._handlers.get(self.next_state, self.idle)()

    def idle(self):
        if self.current_state != "idle":
            self.status_message = "Idle - waiting for input."
        self.current_state = "idle"

    def direct_control(self):
        if not self.arm.connected or not self.arm.initialized:
            self._go_idle("Direct control unavailable until the arm is initialized.")
            return
        if self.current_state != "direct_control":
            self.status_message = "Direct Control - use jog controls to move the arm."
        self.current_state = "direct_control"

    def estop(self):
        if self.current_state != "estop":
            self.arm.stop()
            self.status_message = "STOP active - clear the arm and click Go Home to recover."
            self.current_state = "estop"

    def initial_pose(self):
        if not self.arm.connected:
            self._go_idle("Arm offline - cannot initialize.")
            return
        self.status_message = "Initializing arm..."
        if not self.arm.initialize():
            self._go_idle("Failed to initialize arm.")
        else:
            self._go_idle("Arm initialized - at initial pose.")

    def sleep_arm(self):
        if not self.arm.connected:
            self._go_idle("Arm offline - cannot send Sleep.")
            return
        self.status_message = "Sleep: returning to home position, motors will shut off..."
        self.arm.sleep()
        self._go_idle("Sleep: home position reached, motors off.")

    def calibrate(self):
        _, message = self.camera.estimate_extrinsics_from_tags()
        self._go_idle(message)

    def add_waypoint(self):
        # TODO: student lab
        # Each waypoint stores joint angles + gripper state together.
        # Two consecutive waypoints can have identical joint angles but different gripper states
        # (e.g. WP: arm at grasp position, gripper open -> next WP: same position, gripper closed).
        # Playback executes each waypoint sequentially: move joints first, then apply gripper state.
        # self._go_idle("Student lab: implement add_waypoint() - record joint angles + gripper state.")
        curr_joint_angles = self.arm.get_joint_angles()
        curr_gripper_state = self.arm.gripper_state

        curr_joint_angles.append(curr_gripper_state)

        self.waypoints.append(curr_joint_angles)
        print(curr_joint_angles)
        self._go_idle("waypoint appended")
        
        

    def clear_waypoints(self):
        # TODO: student lab
        # self._go_idle("Student lab: implement clear_waypoints() - erase all recorded waypoints.")
        self.waypoints.clear()


    def playback_waypoints(self):
        # TODO: student lab
        # For each waypoint: move to joint angles (wait), then apply gripper state (wait), then advance.
        # self._go_idle("Student lab: implement playback_waypoints() - replay waypoints in order.")

        # self.waypoints = [np.deg2rad([0.0, 9.9, 31.8, 0.0, 21.9, 0.0]).tolist(),
        #                   np.deg2rad([0.0, 9.9, 36.8, 0.0, 25.9, 0.0]).tolist(),
        #                   np.deg2rad([0.0, 9.9, 40.8, 0.0, 30.9, 0.0]).tolist(),
        #                   np.deg2rad([0.0, 9.9, 36.8, 0.0, 36.9, 0.0]).tolist()]

        for waypoint in self.waypoints:
            self.arm.set_joint_angles(waypoint[:6])
            if self.arm.gripper_state != waypoint[6]:
                if waypoint[6] == 0:
                    self.arm.open_gripper()
                    time.sleep(3)
                else:
                    self.arm.close_gripper()
                    time.sleep(3)
            
            # if waypoint[6] == 0:
            #     self.arm.
        self._go_idle("finished!")

    # ------------------------------------------------------------------
    # Pick & place
    # ------------------------------------------------------------------
    def pick_place(self):
        # Called every loop while the Pick & Place box is checked: return fast unless there is a click.
        if self.current_state != "pick_place":
            self.current_state = "pick_place"
            self.holding_block = False
            self.camera.new_click = False      # ignore clicks made before the mode was turned on
            self.status_message = "Pick & Place - click a block to pick up."

        if not self.camera.new_click:
            return
        self.camera.new_click = False

        target = self._click_to_world()
        if target is None:
            self.status_message = "No valid depth at that click - click again."
            return

        if not self.holding_block:
            self._pick(target)
        else:
            self._place(target)

    def _click_to_world(self, win=3):
        """Last click -> world (x, y, z) mm. Median depth so one bad pixel can't ruin it."""
        u, v = (int(c) for c in self.camera.last_click)
        depth = self.camera.depth_frame_raw
        h, w = depth.shape
        patch = depth[max(v - win, 0):min(v + win + 1, h), max(u - win, 0):min(u + win + 1, w)]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        K = self.camera.intrinsic_matrix
        Zc = float(np.median(valid)) * self.camera.depth_scale_mm
        Xc = (u - K[0, 2]) * Zc / K[0, 0]
        Yc = (v - K[1, 2]) * Zc / K[1, 1]
        p = self.camera.camera_to_world(np.array([Xc, Yc, Zc]))
        return None if p is None else p[:3]

    def _snap_to_block(self, x, y):
        """Nearest detected block within SNAP_RADIUS_MM of (x, y), or None."""
        best, best_dist = None, SNAP_RADIUS_MM
        for d in getattr(self.camera, "block_detections", []):
            dist = np.hypot(d["world"][0] - x, d["world"][1] - y)
            if dist < best_dist:
                best, best_dist = d, dist
        return best

    def _block_yaw(self, d):
        """Gripper yaw (rad) that closes the jaws square across detected block d.
        Squares repeat every 90 deg, so wrap to [-45, 45) and the wrist turns the short way."""
        return np.radians((d["angle_degree"] + 45.0) % 90.0 - 45.0)

    def _flange_pose(self, x, y, z_grasp, yaw=0.0):
        """Top-down grasp: tool pointing straight down, flange 85 mm above the grasp point."""
        return [x, y, z_grasp + FLANGE_TO_GRASP_MM, np.pi, 0.0, yaw]

    def _solve(self, pose, seed=None):
        """Vendor (SDK) IK on the real arm, our numerical IK in the sim, which has no SDK.
        seed: joint angles of a neighboring pose (e.g. the approach above a grasp), so the
        numerical IK stays on that pose's wrist solution instead of flipping the wrist between them."""
        real = getattr(self.arm, "xarm", None) is not None
        q = self.arm.get_ik_sdk(pose) if real else self.arm.get_ik_numerical(pose, q0=seed)
        return None if q is None else list(q)

    def _move(self, q):
        self.arm.set_joint_angles(q, wait=True)

    def _hover_pose(self, x, y, hover_z, yaw):
        """Flange pose with the grasp point at hover_z (mm) over (x, y), or None for no hover."""
        return None if hover_z is None else self._flange_pose(x, y, hover_z, yaw)

    def _pick(self, target, yaw=None, hover_z=None):
        """yaw overrides the jaw yaw from the detected block. hover_z (grasp point height, mm)
        adds a stop that high above the block before going down and after coming up,
        to carry blocks over tall stacks."""
        x, y, z_top = target
        size = "big" if z_top > 31.5 else "small"   # fallback when no block is detected near the click
        yaw_override, yaw = yaw, 0.0

        d = self._snap_to_block(x, y)
        if d is not None:
            x, y, z_top = (float(v) for v in d["world"][:3])   # block's top-face center
            size = d["size"]
            # squares repeat every 90 deg: wrap to [-45, 45) so the wrist turns the short way
            yaw = self._block_yaw(d)
        if yaw_override is not None:
            yaw = yaw_override

        block_h = BLOCK_H[size]
        z_bottom = z_top - block_h
        z_grasp = max(z_bottom + block_h / 2.0 + GRASP_RAISE_MM,
                      z_bottom + TIP_CLEARANCE_MM + FLANGE_TO_TIP_MM - FLANGE_TO_GRASP_MM)
        z_grasp = min(z_grasp, z_top)                # never above the top face

        print("*******************")
        print("picking up block: ")
        print(f"degree: {np.rad2deg(yaw)}")
        print(f"world p: {(x, y, z_top)}")
        print("*******************")

        grasp = self._flange_pose(x, y, z_grasp, -yaw)
        approach = self._flange_pose(x, y, z_grasp + APPROACH_CLEARANCE_MM, -yaw)
        hover = self._hover_pose(x, y, hover_z, -yaw)

        # Solve everything before moving, so an unreachable pose never stops the arm halfway.
        q_approach = self._solve(approach)
        q_grasp = self._solve(grasp, seed=q_approach)
        q_hover = None if hover is None else self._solve(hover, seed=q_approach)
        if q_approach is None or q_grasp is None or (hover is not None and q_hover is None):
            self.status_message = f"({x:.0f}, {y:.0f}) is unreachable - click another block."
            return

        print(f"Pick: {size} block at ({x:.0f}, {y:.0f}, {z_top:.0f}), yaw {np.degrees(yaw):.0f} deg")
        self.arm.open_gripper()                      # queued: opens on the way to the approach, no dwell
        if q_hover is not None:
            self._move(q_hover)
        self._move(q_approach)                       # above the block
        self._move(q_grasp)                          # straight down
        self.arm.close_gripper(wait=True)
        self._move(q_approach)                       # straight up
        if q_hover is not None:
            self._move(q_hover)

        self.holding_block = True
        self.held_block_h = block_h
        self.held_grasp_above_bottom = z_grasp - z_bottom
        self.held_yaw = yaw
        self.camera.new_click = False                # ignore clicks made while the arm was moving
        self.status_message = "Holding block - click where to place it."

    def _place(self, target, go_home=True, hover_z=None, snap=True):
        """go_home=False stays above the place, for events that sense again from there.
        hover_z: as in _pick, a stop that high before going down. snap=False places exactly
        at target, even next to a block (no stacking on it)."""
        x, y, z_surface = target                     # the clicked point is the surface to put it on
        yaw = self.held_yaw                          # on the table: keep the angle it was picked at
        d = self._snap_to_block(x, y) if snap else None
        if d is not None:                            # clicked near a block -> stack centered on it,
            x, y, z_surface = (float(v) for v in d["world"][:3])
            yaw = self._block_yaw(d)                 # turned to match the block underneath

        # Grasp point when the held block's bottom sits just above the surface.
        z_grasp = z_surface + self.held_grasp_above_bottom + PLACE_DROP_MM

        place = self._flange_pose(x, y, z_grasp, -yaw)
        approach = self._flange_pose(x, y, z_grasp + APPROACH_CLEARANCE_MM, -yaw)

        hover = self._hover_pose(x, y, hover_z, -yaw)

        q_approach = self._solve(approach)
        q_place = self._solve(place, seed=q_approach)
        q_hover = None if hover is None else self._solve(hover, seed=q_approach)
        if q_approach is None or q_place is None or (hover is not None and q_hover is None):
            self.status_message = f"({x:.0f}, {y:.0f}) is unreachable - click another spot."
            return

        print(f"Place: at ({x:.0f}, {y:.0f}, {z_surface:.0f})")
        if q_hover is not None:
            self._move(q_hover)
        self._move(q_approach)
        self._move(q_place)
        time.sleep(PLACE_SETTLE_S)
        self.arm.open_gripper(wait=True)
        self._move(q_approach)
        if go_home:
            self._move(list(Q_DEFAULT))              # out of the camera's view for the next click

        self.holding_block = False
        self.camera.new_click = False
        self.status_message = "Placed - click a block to pick up."

    # ------------------------------------------------------------------
    # Event 1: sort blocks (one block per call, see event1.py)
    # ------------------------------------------------------------------
    def event1(self):
        if self.current_state != "event1":
            if not self.arm.connected or not self.arm.initialized:
                self._go_idle("Initialize the arm before running Event 1.")
                return
            self.current_state = "event1"
            self.holding_block = False
            self.event1_task.reset()
            self.status_message = "Event 1: sorting blocks..."
            return                                   # publish the status before the arm moves

        done_message = self.event1_task.step()
        if done_message:
            self._go_idle(done_message)

    def event1l2(self):
            if self.current_state != "event1l2":
                if not self.arm.connected or not self.arm.initialized:
                    self._go_idle("Initialize the arm before running Event 1.")
                    return
                self.current_state = "event1l2"
                self.holding_block = False
                self.event1l2_task.reset()
                self.status_message = "Event 1 Level 2: sorting blocks..."
                return                                   # publish the status before the arm moves
    
            done_message = self.event1l2_task.step()
            if done_message:
                self._go_idle(done_message)

    # ------------------------------------------------------------------
    # Event 2 Level 3: rainbow lines among distractors (one block per call, see event3_level3.py)
    # ------------------------------------------------------------------
    def event2l3(self):
        if self.current_state != "event2l3":
            if not self.arm.connected or not self.arm.initialized:
                self._go_idle("Initialize the arm before running Event 2.")
                return
            self.current_state = "event2l3"
            self.holding_block = False
            self.event2l3_task.reset()
            self.status_message = "Event 2 Level 3: lining up blocks..."
            return                                   # publish the status before the arm moves

        done_message = self.event2l3_task.step()
        if done_message:
            self._go_idle(done_message)

    def reset_board(self):
        """Sim only: arm to the initial pose, then blocks back to the Event 0 layout."""
        try:
            reset_board_event0(self.arm, self.camera)
            self._go_idle("Board reset.")
        except RuntimeError as exc:
            self._go_idle(f"Reset Board failed: {exc}")

    def check_3d(self):
        _, self.status_message = self.camera.check_3d_point(self.arm.get_ee_pose_sdk())
        self.next_state = self.current_state  # go back to what we were doing (e.g. direct control)

class StateMachineThread(QThread):
    updateStatusMessage = pyqtSignal(str)

    def __init__(self, state_machine, parent=None):
        QThread.__init__(self, parent=parent)
        self.sm = state_machine
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)

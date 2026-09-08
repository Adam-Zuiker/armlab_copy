"""
State machine for the arm lab runtime shell.
"""
import time

from PyQt5.QtCore import QThread, pyqtSignal
# import lite6arm
import numpy as np

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
        }

        self.waypoints = []

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

    def pick_place(self):
        # TODO: student lab
        # Student lab: click an object in the video feed to pick it up and place it.
        # 1. Poll camera.new_click; when True, read camera.last_click for pixel (u, v).
        # 2. camera.image_to_world(u, v) -> world XYZ using live depth + extrinsics.
        # 3. arm.get_ik(pose) -> joint angles; arm.set_joint_angles(...) to move above object.
        # 4. arm.close_gripper() to grasp, move to drop position, arm.open_gripper().
        # State persists until the GUI toggle is unchecked.
        if self.current_state != "pick_place":
            self.status_message = "Pick & Place - click an object in the video feed."
            self.current_state = "pick_place"

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

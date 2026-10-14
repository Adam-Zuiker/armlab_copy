"""Exercise pick/place without connecting to the robot or starting motion."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import state_machine as sm_module
from kinematics import DH_STD, FK_dh, IK_geometric, Q_DEFAULT, get_pose_from_T
from state_machine import StateMachine


class FakeArm:
    def __init__(self):
        self.connected = self.initialized = True
        self.speed_pct = 20
        self.gripper_state = 0
        self.joints = np.array(Q_DEFAULT)
        self.commands = []
        self.ik_poses = []
        self.reject = lambda pose: False
        self.move_ok = True

    def get_joint_angles(self):
        return self.joints.copy()

    def get_ik(self, pose):
        self.ik_poses.append(np.array(pose))
        return None if self.reject(pose) else IK_geometric(DH_STD, pose)

    def set_joint_angles(self, joints, wait=False, speed_pct=None):
        pose = np.array(get_pose_from_T(FK_dh(DH_STD, joints, 6)))
        self.commands.append(("move", pose, wait, speed_pct))
        if self.move_ok:
            self.joints = np.array(joints)
        return self.move_ok

    def move_pose(self, pose, joint_solution, wait=True, speed_pct=10):
        return self.set_joint_angles(joint_solution, wait=wait, speed_pct=speed_pct)

    def open_gripper(self, wait=False):
        self.commands.append(("open",))
        self.gripper_state = 0
        return True

    def close_gripper(self, wait=False):
        self.commands.append(("close",))
        self.gripper_state = 1
        return True

    def has_fault(self):
        return False

    def stop(self):
        self.commands.append(("stop",))


class FakeCamera:
    def __init__(self):
        self.camera_calibrated = True
        self.depth_frame_raw = np.ones((200, 200), np.uint16)
        self.last_click = np.array([50, 50])
        self.new_click = False
        self.point = np.array([300., 0, 38., 1.])
        self.workspace_homography = None
        self.block_detections = []
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.array([[0., -1, 0, 0], [-1., 0, 0, 0],
                                          [0., 0, -1, 100], [0., 0, 0, 1]])

    def image_to_world(self, x, y):
        return self.point.copy()

    def workspace_pixel_to_image(self, x, y):
        return np.array([x, y])


class PickPlaceTests(unittest.TestCase):
    def setUp(self):
        self.arm, self.camera = FakeArm(), FakeCamera()
        self.sm = StateMachine(self.arm, self.camera)
        self.clock = 0.
        self.time_patch = patch.object(sm_module.time, "monotonic", side_effect=lambda: self.clock)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.sm.set_next_state("pick_place")
        self.sm.run()

    def click(self, point=None):
        if point is not None:
            self.camera.point = np.r_[point, 1.]
        self.camera.new_click = True
        self.sm.run()

    def finish(self):
        for _ in range(500):
            if not self.sm._pick_place_actions:
                return
            self.clock += 1.1
            self.sm.run()
        self.fail("Sequence did not finish")

    def test_pick_then_table_place_order_heights_and_vertical_motion(self):
        self.click()
        self.assertEqual(self.arm.commands, [])  # all IK solved before any actuation
        self.finish()
        kinds = [command[0] for command in self.arm.commands]
        self.assertEqual(kinds[0], "open")
        self.assertEqual(kinds.count("close"), 1)
        moves = [c for c in self.arm.commands if c[0] == "move"]
        self.assertLessEqual(len(moves), 4)  # continuous phases, not stopped 5 mm steps
        self.assertTrue(all(c[2] is True and c[3] == 30 for c in moves))
        self.assertAlmostEqual(min(c[1][2] for c in moves), 105, places=5)
        close_index = kinds.index("close")
        before_close = self.arm.commands[close_index-1][1]
        np.testing.assert_allclose(before_close[:3], [300, 0, 105], atol=1e-5)
        for command in self.arm.commands[close_index+1:]:
            if command[0] == "move":
                np.testing.assert_allclose(command[1][:2], [300, 0], atol=1e-5)
        self.assertEqual(self.sm._held_block_size_mm, 38)
        self.assertIn("place surface", self.sm.status_message)
        count = len(self.arm.commands)
        self.sm.run()
        self.assertEqual(len(self.arm.commands), count)  # waits for the second click
        self.click([250, 50, 0])
        self.finish()
        place_commands = self.arm.commands[count:]
        release_index = [c[0] for c in place_commands].index("open")
        np.testing.assert_allclose(place_commands[release_index-1][1][:3], [250, 50, 110], atol=1e-5)
        self.assertIsNone(self.sm._held_block_size_mm)
        self.assertIn("click a block", self.sm.status_message)

    def test_place_on_stack_keeps_fingers_and_carried_block_clear(self):
        self.click()
        self.finish()
        count = len(self.arm.commands)
        previous_xy = np.array(get_pose_from_T(FK_dh(DH_STD, self.arm.joints, 6)))[:2]
        self.click([250, 50, 76])
        self.finish()
        commands = self.arm.commands[count:]
        release_index = [c[0] for c in commands].index("open")
        np.testing.assert_allclose(commands[release_index-1][1][:3], [250, 50, 76+110], atol=1e-5)
        for command in commands:
            if command[0] == "move":
                if not np.allclose(command[1][:2], previous_xy):
                    self.assertGreater(command[1][2]-105, 76+sm_module.PICK_CLEARANCE_MM-1e-4)
                previous_xy = command[1][:2]

    def test_biased_depth_does_not_push_fingers_into_table_or_support_block(self):
        self.camera.workspace_homography = np.eye(3)
        self.camera._get_detector_geometry = lambda: {"height_per_depth": np.ones((200, 200))}
        for size in (25, 38):
            with self.subTest(block_size=size), patch.object(sm_module, "PICK_DEFAULT_BLOCK_SIZE_MM", size):
                self.camera.block_height_mm = np.full((200, 200), size, dtype=float)
                self.click([300, 0, size-13])  # axial depth bias lowered absolute world Z
                self.finish()
                for support_height in (0, 38):
                    hanging = self.sm._held_block_hanging_mm
                    self.camera.block_height_mm[:] = support_height
                    count = len(self.arm.commands)
                    self.click([250, 50, support_height-13])
                    self.finish()
                    commands = self.arm.commands[count:]
                    release_index = [c[0] for c in commands].index("open")
                    flange_z = commands[release_index-1][1][2]
                    self.assertGreaterEqual(flange_z-100, support_height+5-1e-5)
                    self.assertGreaterEqual(flange_z-hanging, support_height+5-1e-5)
                    if support_height == 0:
                        self.camera.block_height_mm[:] = size
                        self.click([300, 0, size-13])
                        self.finish()

    def test_ik_failure_anywhere_in_descent_sends_no_commands(self):
        self.arm.reject = lambda pose: pose[2] < 130
        self.click()
        self.assertEqual(self.arm.commands, [])
        self.assertFalse(self.sm._pick_place_actions)
        self.assertEqual(self.sm.next_state, "pick_place")
        self.assertIn("IK", self.sm.status_message)
        self.arm.reject = lambda pose: False
        self.click()
        self.finish()
        self.assertEqual(self.sm._held_block_size_mm, 38)

    def test_unreachable_place_leaves_gripper_closed_and_arm_still(self):
        self.click()
        self.finish()
        count = len(self.arm.commands)
        self.click([1000, 0, 0])
        self.assertEqual(len(self.arm.commands), count)
        self.assertEqual(self.arm.gripper_state, 1)
        self.assertEqual(self.sm._held_block_size_mm, 38)
        self.assertIn("IK", self.sm.status_message)

    def test_gripper_dwell_prevents_lift_and_ignores_busy_click(self):
        self.click()
        while self.sm._pick_place_actions[0][0] != "close":
            self.clock += 1.1
            self.sm.run()
        self.sm.run()  # close
        count = len(self.arm.commands)
        self.camera.new_click = True
        self.sm.run()  # starts dwell
        self.assertFalse(self.camera.new_click)
        self.clock += 0.5
        self.sm.run()
        self.assertEqual(len(self.arm.commands), count)
        self.clock += 0.6
        self.sm.run()  # finishes dwell
        self.sm.run()  # first lift
        self.assertEqual(self.arm.commands[-1][0], "move")
        self.finish()

    def test_bad_depth_or_uncalibrated_click_never_moves(self):
        self.camera.depth_frame_raw[50, 50] = 0
        self.click()
        self.assertIn("depth", self.sm.status_message)
        self.camera.depth_frame_raw[50, 50] = 1
        self.camera.camera_calibrated = False
        self.click()
        self.assertIn("calibrate", self.sm.status_message)
        self.assertEqual(self.arm.commands, [])

    def test_missing_flattened_height_does_not_fall_back_to_biased_depth(self):
        self.camera.workspace_homography = np.eye(3)
        self.camera.block_height_mm = np.full((200, 200), np.nan)
        self.camera._get_detector_geometry = lambda: {"height_per_depth": np.ones((200, 200))}
        self.click([300, 0, 25])
        self.assertEqual(self.arm.commands, [])
        self.assertIn("no valid surface depth", self.sm.status_message)

    def test_detected_small_block_uses_center_size_and_world_yaw(self):
        self.camera.workspace_homography = np.eye(3)
        self.camera.block_detections = [{"world_position_mm": (300, 0, 25), "size_mm": 25,
                                        "orientation_deg": 17., "workspace_center_px": (50, 50),
                                        "rectangle_px": cv2.boxPoints(((50, 50), (25, 25), 17))}]
        self.click([305, 5, 25])
        self.finish()
        self.assertEqual(self.sm._held_block_size_mm, 25)
        self.assertAlmostEqual(self.sm._held_block_yaw, np.deg2rad(-17))
        close_index = [c[0] for c in self.arm.commands].index("close")
        grasp = self.arm.commands[close_index-1][1]
        np.testing.assert_allclose(grasp[:3], [300, 0, 105], atol=1e-5)
        self.assertAlmostEqual(grasp[5], np.deg2rad(-17))

    def test_motion_failure_stops_before_grasp(self):
        self.click()
        self.arm.move_ok = False
        self.finish()
        self.assertEqual(self.sm.next_state, "estop")
        self.assertEqual(self.arm.commands[-1][0], "stop")
        self.assertNotIn("close", [c[0] for c in self.arm.commands])

    def test_controller_fault_reports_code_before_any_pick_motion(self):
        self.arm.has_fault = lambda: True
        self.arm.get_fault_message = lambda: "Controller error 23 (Joints Angle Exceed Limit)."
        self.sm.run()
        self.assertIn("error 23", self.sm.status_message)
        self.assertEqual(self.sm.next_state, "estop")
        self.assertFalse(self.arm.initialized)
        self.assertEqual(self.arm.commands, [("stop",)])

    def test_cancel_discards_remaining_sequence(self):
        self.click()
        self.sm.run()  # opens gripper
        count = len(self.arm.commands)
        self.sm.set_next_state("idle")
        self.sm.run()
        self.assertFalse(self.sm._pick_place_actions)
        self.assertEqual(len(self.arm.commands), count)


class ArmMotionContractTests(unittest.TestCase):
    def test_real_pose_uses_linear_motion_and_composes_tcp_offset(self):
        from lite6arm import Lite6Arm
        arm = Lite6Arm.__new__(Lite6Arm)
        arm.connected = arm.initialized = True
        arm.speed_pct = 20
        arm.xarm = Mock()
        arm.xarm.has_error = False
        arm.xarm.tcp_offset = [0, 0, 100, 0, 0, 0]
        arm.xarm.set_position.return_value = 0
        self.assertTrue(arm.move_pose([300, 0, 110, np.pi, 0, 0], np.zeros(6), wait=True))
        kwargs = arm.xarm.set_position.call_args.kwargs
        self.assertTrue(kwargs["wait"])
        self.assertTrue(kwargs["is_radian"])
        self.assertIsNone(kwargs["radius"])
        self.assertAlmostEqual(kwargs["z"], 10)
        arm.xarm.set_servo_angle.assert_not_called()

    def test_sim_pose_streams_cartesian_line_and_settles_only_at_end(self):
        from lite6arm_sim import SimArm
        arm = SimArm.__new__(SimArm)
        arm.connected = arm.initialized = True
        arm.speed_pct = 20
        arm.mvacc_rad_s2 = 10
        arm._motion_stop_generation = 0
        arm.get_joint_angles = lambda: Q_DEFAULT
        arm.get_ik = lambda pose: IK_geometric(DH_STD, pose)
        arm._send_control = Mock()
        arm.set_joint_angles = Mock(return_value=True)
        start = np.array(get_pose_from_T(FK_dh(DH_STD, Q_DEFAULT, 6)))
        goal = start.copy()
        goal[2] += 30
        solution = IK_geometric(DH_STD, goal)
        with patch("lite6arm_sim.time.monotonic", side_effect=[0., 0., 0.5, 100.]), patch("lite6arm_sim.time.sleep"):
            self.assertTrue(arm.move_pose(goal, solution))
        self.assertEqual(arm._send_control.call_count, 3)
        for call in arm._send_control.call_args_list:
            actual = np.array(get_pose_from_T(FK_dh(DH_STD, call.kwargs["qj_pos"], 6)))
            np.testing.assert_allclose(actual[:2], start[:2], atol=1e-5)
        self.assertTrue(arm.set_joint_angles.call_args.kwargs["wait"])
        self.assertEqual(arm.set_joint_angles.call_count, 1)

    def test_joint_six_fault_includes_underlying_servo_diagnostics(self):
        from lite6arm import Lite6Arm
        arm = Lite6Arm.__new__(Lite6Arm)
        arm.connected = True
        arm.xarm = Mock()
        arm.xarm.state = 4
        arm.xarm.get_err_warn_code.return_value = (0, [16, 0])
        servos = [
            {"servo_id": 1, "code": 0, "status": 0, "title": "Normal"},
            {"servo_id": 6, "code": 23, "status": 0, "title": "Test servo fault"},
        ]
        for status in (0, 1):
            with self.subTest(sdk_status=status):
                arm.xarm.get_servo_debug_msg.return_value = (status, servos)
                message = arm.get_fault_message()
                self.assertIn("Controller error 16", message)
                self.assertIn("J6 servo error 23 (Test servo fault)", message)
                self.assertNotIn("unavailable", message)
        arm.xarm.clean_error.assert_not_called()
        arm.xarm.motion_enable.assert_not_called()

    def test_servo_query_error_without_data_remains_unavailable(self):
        from lite6arm import Lite6Arm
        arm = Lite6Arm.__new__(Lite6Arm)
        arm.connected = True
        arm.xarm = Mock()
        arm.xarm.state = 4
        arm.xarm.get_err_warn_code.return_value = (0, [16, 0])
        arm.xarm.get_servo_debug_msg.return_value = (1, [])
        message = arm.get_fault_message()
        self.assertIn("Controller error 16", message)
        self.assertIn("Servo diagnostics unavailable", message)

    def test_real_initialization_failure_does_not_claim_arm_is_ready(self):
        from lite6arm import Lite6Arm
        arm = Lite6Arm.__new__(Lite6Arm)
        arm.connected = arm.initialized = True
        arm.mvacc_rad_s2 = 10
        arm.xarm = Mock()
        arm.xarm.has_error = False
        arm.xarm.set_servo_angle.return_value = -2
        arm.xarm.get_err_warn_code.return_value = (0, [23, 0])
        arm.xarm.state = 4
        self.assertFalse(arm.initialize())
        self.assertFalse(arm.initialized)
        self.assertIn("error 23", arm.get_fault_message())

    def test_real_wrapper_wait_flag_speed_cap_and_gripper_failure(self):
        from lite6arm import Lite6Arm
        arm = Lite6Arm.__new__(Lite6Arm)  # no network connection
        arm.connected = arm.initialized = True
        arm.speed_pct = 20
        arm.mvacc_rad_s2 = 10
        arm.gripper_state = 0
        arm.xarm = Mock()
        arm.xarm.has_error = False
        arm.xarm.set_servo_angle.return_value = 0
        self.assertTrue(arm.set_joint_angles([0]*6, wait=True, speed_pct=10))
        kwargs = arm.xarm.set_servo_angle.call_args.kwargs
        self.assertTrue(kwargs["wait"])
        self.assertAlmostEqual(kwargs["speed"], np.pi*0.1)
        arm.xarm.close_lite6_gripper.return_value = -1
        self.assertFalse(arm.close_gripper())
        self.assertEqual(arm.gripper_state, 0)

    def test_sim_wait_requires_settled_feedback(self):
        from lite6arm_sim import SimArm
        arm = SimArm.__new__(SimArm)  # no LCM connection
        arm.connected = arm.initialized = True
        arm.speed_pct = 20
        arm.mvacc_rad_s2 = 10
        arm._motion_stop_generation = 0
        arm._send_control = Mock()
        goal = np.ones(6)*0.1
        arm.get_joint_angles = Mock(side_effect=[np.zeros(6), goal, goal, goal])
        with patch("lite6arm_sim.time.sleep"):
            self.assertTrue(arm.set_joint_angles(goal, wait=True, speed_pct=10))
        self.assertEqual(arm.get_joint_angles.call_count, 4)
        self.assertAlmostEqual(arm._send_control.call_args.kwargs["speed"], np.pi*0.1)

    def test_sim_wait_timeout_stops_and_returns_failure(self):
        from lite6arm_sim import SimArm
        arm = SimArm.__new__(SimArm)
        arm.connected = arm.initialized = True
        arm.speed_pct = 20
        arm.mvacc_rad_s2 = 10
        arm._motion_stop_generation = 0
        arm._send_control = Mock()
        arm.stop = Mock()
        with patch("lite6arm_sim.time.monotonic", side_effect=[0., 31.]):
            self.assertFalse(arm.set_joint_angles(np.zeros(6), wait=True))
        arm.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()

import time
from collections import namedtuple

import mujoco
import numpy as np

from .lcm2mujoco_bridge import Lcm2MujocoBridge
from lite6_sim.lcm_msgs import (sim_reset_t, sim_world_t, ufactory_lite6_display_t,
                                ufactory_lite6_gripper_t)
from lite6_sim.utils import *

# The model also holds a gripper and free blocks, so nothing may assume that
# qpos / qvel / ctrl are the arm alone. Everything arm-related goes through
# these names and the index arrays built from them.
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
ARM_ACTUATORS = tuple(f"joint{i}_motor" for i in range(1, 7))
NUM_ARM_JOINTS = len(ARM_JOINTS)

# ufactory_lite6_control_t.mode. STOP and RESUME are the real arm's stop state and
# set_state(0): a stop holds the arm until a resume, whichever client sent it.
MODE_POSITION, MODE_JOINT_VEL, MODE_CART_VEL, MODE_STOP, MODE_RESUME = range(5)

GRIPPER_TRAVEL_TIME_S = 0.4   # jaw target ramp, fully open <-> fully closed
WORLD_PUBLISH_DT_S = 0.02     # sim_world_t rate (sim time), 50 Hz

# A block of armlab_scene.xml: a free body named block_<size>_<color> with one box geom.
SimBlock = namedtuple("SimBlock", "body color edge qadr vadr")
BLOCKS_ONLY = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # ray casts: group 0, the blocks


class UfactoryLite6Bridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        launch_args = getattr(config, "launch_args", None)
        in_replay_mode = bool(getattr(launch_args, "replay", False))
        if in_replay_mode:
            config.lcm_udp_multicast_group = "udpm://239.255.76.67:7667?ttl=0"
        super().__init__(mj_model, mj_data, config)
        self.in_replay_mode = in_replay_mode
        self.ee_body_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "link6")

        # Arm joints and actuators by name. The named views raise on a typo, where
        # mj_name2id would return -1 and quietly index the last body's joint.
        jnt_ids = [self.mj_model.joint(n).id for n in ARM_JOINTS]
        self.arm_qpos_adr = self.mj_model.jnt_qposadr[jnt_ids]
        self.arm_dof_adr = self.mj_model.jnt_dofadr[jnt_ids]
        self.arm_act_ids = np.array([self.mj_model.actuator(n).id for n in ARM_ACTUATORS])
        arm_jnt_range = self.mj_model.jnt_range[jnt_ids]

        # Internal desired trajectory state. Shaped by the incoming speed/mvacc limits
        # and written straight to mj_data.ctrl, which the position actuators servo to.
        self._q_des = None                          # desired joint position (6,), set on the first step
        self._qd_des = np.zeros(NUM_ARM_JOINTS)     # desired joint velocity (6,)
        self._target = None                         # position-mode target being tracked
        self._limit_scale = np.ones(NUM_ARM_JOINTS) # per-joint share of the current move
        self.motion_done = True
        self.stopped = False                        # stop latch: MODE_STOP until MODE_RESUME

        # Cartesian jog (mode 2): velocity IK on a scratch MjData, so the Jacobian is
        # evaluated at the commanded configuration and the solver integrates its own
        # state instead of chasing sensor noise.
        self._ik_solver = DifferentialIKSolver(arm_jnt_range[:, 0], arm_jnt_range[:, 1])
        self._ik_data = mujoco.MjData(self.mj_model)

        # Gripper: the latest binary command, held with no watchdog like the real
        # jaws, and the ramped actuator target that realises it.
        self.gripper_act_id = self.mj_model.actuator("gripper_motor").id
        self._gripper_cmd = ufactory_lite6_gripper_t.OPEN
        self._gripper_goal = None
        self._gripper_ctrl = float(self.mj_model.actuator_ctrlrange[self.gripper_act_id, 0])
        self.lc.subscribe(config.robot_gripper_topic, self.lcm_gripper_handler)

        # Blocks, and the keyframe a reset asks for (set on the LCM thread, applied on
        # the sim thread).
        self.blocks = []
        for jnt in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
                body = int(self.mj_model.jnt_bodyid[jnt])
                edge = 2.0 * float(self.mj_model.geom_size[self.mj_model.body_geomadr[body], 0])
                self.blocks.append(SimBlock(body, self.mj_model.body(body).name.split("_")[-1], edge,
                                            int(self.mj_model.jnt_qposadr[jnt]), int(self.mj_model.jnt_dofadr[jnt])))
        print(f"=> blocks: {len(self.blocks)}")
        self._reset_keyframe = None
        self._next_world_pub_time = 0.0
        self.topic_world = config.world_topic
        self.lc.subscribe(config.reset_topic, self.lcm_reset_handler)

        # Display overlays driven by the upper-level GUI over the display msg LCM channel.
        # Only the raw values are stored here; main.py owns the rendering so the
        # scene geometry is built on the viewer thread rather than the LCM thread.
        self.vis_ghost_q = np.zeros(6)              # ghost arm joint angles (rad)
        self.vis_fk_pos = np.zeros(3)               # FK marker position (m, world)
        self.vis_fk_R = np.eye(3)                   # FK marker orientation
        self.vis_fk_box_size = [0.025, 0.025, 0.025]
        self.lc.subscribe(config.robot_display_topic, self.lcm_display_handler)

    def num_joint_motors(self):
        return len(ARM_ACTUATORS)

    def lcm_display_handler(self, channel, data):
        # Upper-level display request: ghost arm (IK preview) and FK pose marker.
        msg = ufactory_lite6_display_t.decode(data)
        self.vis_ghost = bool(msg.show_ghost)
        if self.vis_ghost:
            self.vis_ghost_q = np.array(msg.ghost_qj_pos, dtype=float)
        self.vis_fk = bool(msg.show_fk)
        if self.vis_fk:
            self.vis_fk_pos = np.array(msg.fk_pos, dtype=float)
            # rpy_to_quat/quat_to_rot use the same fixed-axis RPY convention as
            # the upper level's get_pose_from_T, so no extra conversion is needed.
            self.vis_fk_R = quat_to_rot(rpy_to_quat(np.array(msg.fk_rpy, dtype=float)))

    def lcm_gripper_handler(self, channel, data):
        msg = ufactory_lite6_gripper_t.decode(data)
        self._gripper_cmd = int(msg.command)

    def lcm_reset_handler(self, channel, data):
        self._reset_keyframe = sim_reset_t.decode(data).keyframe

    def _apply_pending_reset(self):
        # Every block to where the keyframe puts it, at rest. The arm is left alone.
        name, self._reset_keyframe = self._reset_keyframe, None
        if name is None:
            return
        try:
            key_qpos = self.mj_model.key(name).qpos
        except KeyError:
            print(f"=> reset: no keyframe '{name}'")
            return
        for b in self.blocks:
            self.mj_data.qpos[b.qadr:b.qadr + 7] = key_qpos[b.qadr:b.qadr + 7]
            self.mj_data.qvel[b.vadr:b.vadr + 6] = 0.0
            self.mj_data.qacc_warmstart[b.vadr:b.vadr + 6] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _seen_from_above(self, block):
        # What a camera looking straight down sees: a block on the board (centre above
        # z = 0, unlike the parked ones) with no other block above its centre.
        pos = self.mj_data.xpos[block.body]
        if pos[2] <= 0.0:
            return False
        geom = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(self.mj_model, self.mj_data, pos + [0.0, 0.0, 1.0], np.array([0.0, 0.0, -1.0]),
                      BLOCKS_ONLY, 0, -1, geom)
        return geom[0] >= 0 and self.mj_model.geom_bodyid[geom[0]] == block.body

    def _block_yaw(self, block):
        # A cube resting on any face: the body axis closest to vertical points up, and
        # the yaw is the heading of the next axis, wrapped to [-pi/4, pi/4).
        R = self.mj_data.xmat[block.body].reshape(3, 3)
        side = (int(np.argmax(np.abs(R[2]))) + 1) % 3
        yaw = np.arctan2(R[1, side], R[0, side])
        return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    def _publish_world_state(self):
        # Ground truth the real arm and camera cannot report, at a reduced rate.
        if self.mj_data.time < self._next_world_pub_time:
            return
        self._next_world_pub_time = self.mj_data.time + WORLD_PUBLISH_DT_S
        msg = sim_world_t()
        msg.timestamp = time.time_ns()
        msg.stopped = self.stopped
        msg.motion_done = bool(self.motion_done)
        msg.q_des = list(self._q_des) if self._q_des is not None else [0.0] * NUM_ARM_JOINTS
        seen = [b for b in self.blocks if self._seen_from_above(b)]
        msg.num_blocks = len(seen)
        msg.color = [b.color for b in seen]
        msg.size = [b.edge for b in seen]
        msg.pos = [self.mj_data.xpos[b.body].tolist() for b in seen]
        msg.yaw = [self._block_yaw(b) for b in seen]
        self.lc.publish(self.topic_world, msg.encode())

    def parse_robot_specific_low_state(self):
        dof = self.arm_dof_adr
        # Joint-space inertia matrix
        temp_inertia_mat = np.zeros((self.mj_model.nv, self.mj_model.nv))
        mujoco.mj_fullM(self.mj_model, temp_inertia_mat, self.mj_data.qM)
        self.low_state.inertia_mat = temp_inertia_mat[np.ix_(dof, dof)].tolist()
        self.low_state.bias_force = self.mj_data.qfrc_bias[dof].tolist()

        # End-effector translational Jacobian and its time-derivative-times-qvel
        ee_pos = self.mj_data.xpos[self.ee_body_id]
        J_ee = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_ee, None, ee_pos, self.ee_body_id)

        dJ_ee = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_ee, None, ee_pos, self.ee_body_id)
        dJdq_ee = dJ_ee[:, dof] @ self.low_state.qj_vel

        self.low_state.J_ee = J_ee[:, dof].tolist()
        self.low_state.dJdq_ee = dJdq_ee.tolist()
        self.low_state.p_ee = ee_pos.tolist()

    @staticmethod
    def _rate_limit(v_prev, v_target, accel_limit, dt):
        # Move v_prev toward v_target with per-element |delta v| <= accel_limit*dt.
        # accel_limit is one value for all joints or one per joint.
        v_prev = np.asarray(v_prev, dtype=float)
        v_target = np.asarray(v_target, dtype=float)
        if np.all(np.asarray(accel_limit) <= 0.0) or dt <= 0.0:
            return v_target.copy()  # no accel limit -> jump straight to target velocity
        step = np.asarray(accel_limit, dtype=float) * dt
        dv = np.clip(v_target - v_prev, -step, step)
        return v_prev + dv

    def _servo_profile(self, q_des, qd_des, q_target, v_max, a_max, dt):
        # Trapezoidal (velocity/accel-limited) follower toward q_target. v_max and a_max
        # are one value for all joints or one per joint. Returns the next (q_des, qd_des)
        # and decelerates so it stops cleanly at the target.
        q_des = np.asarray(q_des, dtype=float)
        qd_des = np.asarray(qd_des, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        err = q_target - q_des
        # Fastest speed from which we can still brake to a stop within |err|.
        v_stop = np.sqrt(2.0 * np.maximum(a_max, 1e-9) * np.abs(err))
        v_cmd = np.sign(err) * np.minimum(v_max, v_stop)
        qd_new = self._rate_limit(qd_des, v_cmd, a_max, dt)
        q_new = q_des + qd_new * dt
        # Snap when essentially there to avoid dithering around the target.
        close = np.abs(q_target - q_new) < 1e-4
        q_new = np.where(close, q_target, q_new)
        qd_new = np.where(close, 0.0, qd_new)
        return q_new, qd_new

    def _position_step(self, target, speed, mvacc, dt):
        # Position / servo mode. speed <= 0 tracks the target directly (backward compatible).
        if speed <= 0.0:
            self._target = target
            return target.copy(), np.zeros(NUM_ARM_JOINTS)
        if self._target is None or np.any(target != self._target):
            # A new target. From rest, give each joint the limits scaled by its share of
            # the move: every joint then runs the same profile scaled down, so all arrive
            # together, as with set_servo_angle. Redirected mid-move, use full limits.
            dist = np.abs(target - self._q_des)
            from_rest = not np.any(self._qd_des) and dist.max() > 0.0
            self._limit_scale = dist / dist.max() if from_rest else np.ones(NUM_ARM_JOINTS)
            self._target = target
        s = self._limit_scale
        return self._servo_profile(self._q_des, self._qd_des, target, speed * s, mvacc * s, dt)

    def _ee_jacobian(self, q):
        # 6 x 6 end-effector Jacobian [v; w] over the arm joints at configuration q, in
        # the world frame, plus the end-effector rotation. Evaluated on the scratch
        # MjData so the live simulation state is untouched.
        self._ik_data.qpos[self.arm_qpos_adr] = q
        mujoco.mj_kinematics(self.mj_model, self._ik_data)
        mujoco.mj_comPos(self.mj_model, self._ik_data)
        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))
        ee_pos = self._ik_data.xpos[self.ee_body_id]
        mujoco.mj_jac(self.mj_model, self._ik_data, jacp, jacr, ee_pos, self.ee_body_id)
        R_ee = np.array(self._ik_data.xmat[self.ee_body_id]).reshape(3, 3)
        return np.vstack([jacp[:, self.arm_dof_adr], jacr[:, self.arm_dof_adr]]), R_ee

    def _cartesian_step(self, cmd, dt):
        # Cartesian jog: map the commanded end-effector twist to joint velocities by
        # velocity IK, then integrate. This is the sim's stand-in for the real arm's
        # firmware mode 5, so the upper level only ever streams a twist.
        J, R_ee = self._ee_jacobian(self._q_des)
        twist = np.asarray(cmd.v_ee, dtype=float)
        if bool(getattr(cmd, "ee_frame", False)):
            # Tool frame -> world. Linear and angular blocks rotate the same way.
            twist = np.hstack([R_ee @ twist[:3], R_ee @ twist[3:]])

        qd_target = self._ik_solver.solve(J, twist, self._q_des, dt, qd_limit=cmd.speed)
        if qd_target is None:
            qd_target = np.zeros(NUM_ARM_JOINTS)  # unsolvable: hold instead of guessing

        # Same accel limiting as the joint jog, so starts and stops are not steps.
        qd_des = self._rate_limit(self._qd_des, qd_target, cmd.mvacc, dt)
        return self._q_des + qd_des * dt, qd_des

    def update_gripper_cmd(self):
        # Ramp the jaw target toward the commanded end. The position actuator's
        # forcerange caps the squeeze once the jaws meet a block. STOP freezes the
        # target where the jaws are, like cutting the drive.
        lo, hi = self.mj_model.actuator_ctrlrange[self.gripper_act_id]
        cmd = self._gripper_cmd
        if cmd == ufactory_lite6_gripper_t.CLOSE:
            self._gripper_goal = hi
        elif cmd == ufactory_lite6_gripper_t.OPEN:
            self._gripper_goal = lo
        elif self._gripper_goal is not None:
            self._gripper_goal = None
            self._gripper_ctrl = float(np.clip(self.mj_data.actuator_length[self.gripper_act_id], lo, hi))
        if self._gripper_goal is not None:
            step = (hi - lo) / GRIPPER_TRAVEL_TIME_S * self.dt
            self._gripper_ctrl += float(np.clip(self._gripper_goal - self._gripper_ctrl, -step, step))
        self.mj_data.ctrl[self.gripper_act_id] = self._gripper_ctrl

    def update_motor_cmd(self):
        # Once per sim step, after mj_step: a pending block reset, the gripper and arm
        # targets, then the world state. Only this thread steps the sim, so a reset
        # outside the viewer lock can at worst show one frame mid-reset.
        self._apply_pending_reset()
        self.update_gripper_cmd()
        self._update_arm_cmd()
        self._publish_world_state()

    def _update_arm_cmd(self):
        # Position control: mj_data.ctrl is a joint angle target and the <position>
        # actuators in the XML close the servo loop. All this has to do is shape the
        # desired trajectory (q_des) from the incoming command and its speed/mvacc limits.
        acts = self.arm_act_ids
        lo = self.mj_model.actuator_ctrlrange[acts, 0]   # = joint travel limits
        hi = self.mj_model.actuator_ctrlrange[acts, 1]
        dt = self.dt

        have_live_cmd = True
        if self._lcm_cmd_daemon is not None:
            self._lcm_cmd_daemon.update()
            self._print_lcm_cmd_daemon()
            if self._lcm_cmd_daemon.is_error():
                have_live_cmd = False

        # main.py resets to the keyframe after building the bridge, so start from here.
        if self._q_des is None:
            self._q_des = np.array(self.mj_data.qpos[self.arm_qpos_adr], dtype=float)

        cmd = self._compute_delayed_low_cmd()
        mode = int(cmd.mode)
        if mode == MODE_STOP and not self.stopped:
            # Hold where the arm is. The servos lag the trajectory by about 0.1 s of
            # motion, so holding _q_des would carry the arm on past the stop.
            self.stopped = True
            self._q_des = np.array(self.mj_data.qpos[self.arm_qpos_adr], dtype=float)
        elif mode == MODE_RESUME:
            self.stopped = False
        position_mode = mode in (MODE_POSITION, MODE_RESUME)

        # Hold the last desired pose while stopped, or when the command stream drops.
        if self.stopped or not have_live_cmd:
            self._qd_des = np.zeros(NUM_ARM_JOINTS)
            self._target = None
            self.motion_done = False
            self.mj_data.ctrl[acts] = np.clip(self._q_des, lo, hi)
            return

        if position_mode:
            # RESUME carries the pose to hold: track it directly, like speed 0.
            speed = float(cmd.speed) if mode == MODE_POSITION else 0.0
            target = np.clip(np.asarray(cmd.qj_pos, dtype=float), lo, hi)
            q_des, qd_des = self._position_step(target, speed, float(cmd.mvacc), dt)
        else:
            self._target = None
            if mode == MODE_CART_VEL:
                # Cartesian velocity / jog.
                q_des, qd_des = self._cartesian_step(cmd, dt)
            else:
                # Joint velocity / jog: ramp desired velocity toward cmd.qj_vel under mvacc, integrate.
                qd_target = np.asarray(cmd.qj_vel, dtype=float)
                if cmd.speed > 0.0:
                    qd_target = np.clip(qd_target, -cmd.speed, cmd.speed)
                qd_des = self._rate_limit(self._qd_des, qd_target, cmd.mvacc, dt)
                q_des = self._q_des + qd_des * dt

        self._q_des = np.clip(q_des, lo, hi)
        self._qd_des = qd_des
        self.motion_done = position_mode and np.array_equal(self._q_des, self._target)
        self.mj_data.ctrl[acts] = self._q_des

    def lcm_state_handler(self, channel, data):
        if self.mj_data is None:
            return
        # In replay mode, parse_common_low_state is skipped on the sim thread, so there is no race over low_state.qj_*
        msg = self.low_state_type.decode(data)

        # Update mj_data for visualization
        self.mj_data.qpos[self.arm_qpos_adr] = msg.qj_pos
        self.mj_data.qvel[:] = 0
        self.mj_data.act[:] = False
        self.mj_data.qacc_warmstart[:] = 0
        # ctrl is a joint angle target under position control
        self.mj_data.ctrl[self.arm_act_ids] = msg.qj_pos

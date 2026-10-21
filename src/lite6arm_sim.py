"""
MuJoCo simulation arm.

Instead of talking to the arm hardware, SimArm talks to the mujoco_sim MuJoCo
simulation over LCM. It is the mirror image of the bridge:
    * GET STATE  : subscribes to  ufactory_lite6_state   (ufactory_lite6_state_t)
                   and           ufactory_lite6_world   (sim_world_t, trajectory status)
    * SEND CONTROL: publishes to   ufactory_lite6_control (ufactory_lite6_control_t)
                   and           ufactory_lite6_gripper (ufactory_lite6_gripper_t)
The MuJoCo model is position-controlled (`<position>` actuators, gains in the XML), so
SimArm only ever sends targets -- joint angles, joint velocities, or an end-effector
twist -- and reads sim state back. The bridge shapes those into a joint trajectory:
    * mode 0 : servo to qj_pos, speed/accel limited, all joints arriving together
    * mode 1 : integrate qj_vel (joint jog)
    * mode 2 : velocity IK on v_ee (Cartesian jog), the sim's stand-in for the real arm's firmware mode 5
    * mode 3 / 4 : stop / resume, the real arm's stop state and set_state(0)
SimArm exposes the same public API as Lite6Arm so control_station and state_machine can use either interchangeably (`--sim` vs `--real`).

It also behaves like the real arm where student code can tell the difference:
    * set_joint_angles() queues moves and runs them one after another, like the
      xArm controller's command queue. wait=True blocks until the move is done.
    * Gripper calls with sync=True wait their turn in that queue, sync=False acts
      at once. wait=True sleeps GRIPPER_DWELL_S from the call, as on hardware.
    * stop() latches a stop in the bridge, so the arm holds and ignores motion
      from every client until enable() or initialize().
SimArmThread (GUI) re-sends the latest command every 50 ms, so the bridge keeps
following it.
"""
import os
import queue
import sys
import threading
import time

import numpy as np

# Temp fix: add mujoco_sim to path
_MUJOCO_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_sim")
if _MUJOCO_SIM_DIR not in sys.path:
    sys.path.insert(0, _MUJOCO_SIM_DIR)

from PyQt5.QtCore import QThread, pyqtSignal

from kinematics import (DH_STD, JOINT_LIMITS, Q_DEFAULT, FK_dh, FK_pox, IK_geometric, IK_numerical, M, S_list, get_pose_from_T)

try:
    import lcm
    from lite6_sim.lcm_msgs import (sim_world_t, ufactory_lite6_control_t, ufactory_lite6_display_t,
                                    ufactory_lite6_gripper_t, ufactory_lite6_state_t)
    _LCM_AVAILABLE = True
except Exception as _exc:  # any import failure just disables the sim link
    _LCM_AVAILABLE = False
    _LCM_IMPORT_ERROR = _exc

STATE_CHANNEL = "ufactory_lite6_state"       # bridge -> us (sim state)
WORLD_CHANNEL = "ufactory_lite6_world"       # bridge -> us (trajectory status, stop latch)
CONTROL_CHANNEL = "ufactory_lite6_control"   # us -> bridge (motion command)
GRIPPER_CHANNEL = "ufactory_lite6_gripper"   # us -> bridge (jaw command)
DISPLAY_CHANNEL = "ufactory_lite6_display"   # us -> bridge (viewer overlays only)
LCM_URL = "udpm://239.255.76.67:7667?ttl=0"  # ttl=0: stays on this host, so stations do not cross-talk

# Control modes understood by the bridge (see ufactory_lite6_control_t.lcm).
MODE_POSITION = 0    # servo to qj_pos
MODE_JOINT_VEL = 1   # integrate qj_vel (joint jog)
MODE_CART_VEL = 2    # velocity IK on v_ee (Cartesian jog)
MODE_STOP = 3        # hold, ignore motion until MODE_RESUME
MODE_RESUME = 4      # clear a stop and hold qj_pos

# Full-scale Cartesian jog rates, before the speed slider scales them. These match
# the caps the real arm's mode 5 accepts (lite6arm.start_cartesian_jog), so the two
# arms jog at the same speed for the same slider setting.
CART_JOG_LIN_MM_S = 150.0
CART_JOG_ROT_RAD_S = 1.0

GRIPPER_DWELL_S = 0.8                   # same dwell as the real Gripper Lite path (lite6arm.py)
INIT_SPEED_RAD_S = 0.1 * np.pi          # initialize() and sleep() speed, as on the real arm
MIN_MOVE_SPEED_RAD_S = 0.01 * np.pi     # the speed slider's 1 % floor. Speed 0 would make the bridge jump.

# A move is done when the bridge's trajectory has reached the target (within
# SETTLE_POS_RAD, wider than the rounding of the MJCF joint limits) and the joints
# have caught up. Contact can hold a joint off target, e.g. a block under the
# jaws, so after SETTLE_GIVE_UP_S at the target the move counts as done anyway,
# as the real controller reports done when its trajectory ends.
SETTLE_POS_RAD = 1e-3
SETTLE_VEL_RAD_S = 0.02
SETTLE_GIVE_UP_S = 1.0
# stop() and enable() wait this long for the bridge to report its stop latch changed.
LATCH_TIMEOUT_S = 0.5


class _QueueItem:
    """One entry of the emulated controller queue: action(epoch, *args) -> bool."""

    def __init__(self, epoch, action, args):
        self.epoch = epoch
        self.action = action
        self.args = args
        self.done = threading.Event()
        self.ok = False


class SimArm:
    """Simulation arm interface; drives the MuJoCo sim over LCM."""

    def __init__(self):
        self.connected = False
        self.initialized = False
        self.num_joints = 6
        self.joint_angles = [0.0] * self.num_joints
        # Lite 6 spec limits. The real arm reads these from firmware; the sim
        # has no firmware to query, so it uses the table in kinematics.py.
        self.joint_limits = [(lo, hi) for lo, hi in JOINT_LIMITS]
        self.dh_params = DH_STD  # standard Lite 6 DH table (sim has no firmware to query)
        self.speed_pct = 20.0
        self.mvacc_rad_s2 = 10.0

        # LCM: subscribe to sim state, publish joint control.
        self._state_lock = threading.Lock()   # guards _latest_state, _latest_world
        self._pub_lock = threading.Lock()      # guards _latest_cmd + serializes publish
        self._latest_state = None              # last ufactory_lite6_state_t
        self._latest_world = None              # last sim_world_t
        self._latest_cmd = None                # last ufactory_lite6_control_t (for re-streaming)

        # Emulated controller queue. Flushing moves _epoch on, and the worker drops
        # anything from an older epoch, including the move it is running. _cmd_lock is
        # held across "epoch check + publish", so a flushed move is never sent after
        # the command that flushed it.
        self._queue = queue.Queue()
        self._epoch = 0
        self._cmd_lock = threading.Lock()

        # Viewer overlay state (published on DISPLAY_CHANNEL, no effect on the arm).
        self._ghost_q = None                   # IK preview joint angles (rad), None = hidden
        self._show_fk = True                   # draw the FK pose marker
        self._lc = None
        self._threads = []
        self._running = False
        if _LCM_AVAILABLE:
            try:
                self._lc = lcm.LCM(LCM_URL)
                self._lc.subscribe(STATE_CHANNEL, self._on_state)
                self._lc.subscribe(WORLD_CHANNEL, self._on_world)
                self._running = True
                for target in (self._lcm_loop, self._queue_worker):
                    thread = threading.Thread(target=target, daemon=True)
                    thread.start()
                    self._threads.append(thread)
                self.connected = True
            except Exception as exc:
                print(f"WARNING: SimArm LCM unavailable ({exc})")
                self._lc = None
        else:
            print("WARNING: lite6_sim/lcm not importable. Sim arm disabled")

    def set_speed_pct(self, pct):
        self.speed_pct = float(np.clip(pct, 0, 100))

    def _speed_rad_s(self):
        # Joint speed (rad/s) from the speed slider, matching the real arm's mapping.
        return self.speed_pct / 100.0 * np.pi

    # --- LCM GET STATE (receiver) ---

    def _lcm_loop(self):
        while self._running:
            try:
                self._lc.handle_timeout(100)
            except Exception:
                pass

    def _on_state(self, channel, data):
        try:
            msg = ufactory_lite6_state_t.decode(data)
        except Exception:
            return
        with self._state_lock:
            self._latest_state = msg

    def _on_world(self, channel, data):
        try:
            msg = sim_world_t.decode(data)
        except Exception:
            return
        with self._state_lock:
            self._latest_world = msg

    def _latest(self):
        with self._state_lock:
            return self._latest_state, self._latest_world

    def _stopped(self):
        # The bridge's stop latch, set by a stop() from any client.
        world = self._latest()[1]
        return world is not None and world.stopped

    # --- LCM SEND CONTROL (publisher). Each command method calls _send_control. ---

    def _send_control(self, qj_pos=None, qj_vel=None, v_ee=None, ee_frame=False,
                      mode=MODE_POSITION, speed=0.0, mvacc=0.0):
        # Build and publish one control command, and store it for re-streaming.
        # mode selects which of qj_pos / qj_vel / v_ee the bridge follows.
        # speed/mvacc: trajectory-shaping limits consumed by the bridge (rad/s, rad/s^2).
        # qj_tau/kp/kd stay zero: the sim is position-controlled, the servo gains live in the XML.
        if self._lc is None:
            return
        z = [0.0] * self.num_joints
        msg = ufactory_lite6_control_t()
        msg.timestamp = time.time_ns()
        msg.mode = int(mode)
        msg.speed = float(speed)
        msg.mvacc = float(mvacc)
        msg.qj_tau = list(z)
        msg.qj_pos = list(qj_pos) if qj_pos is not None else list(z)
        msg.qj_vel = list(qj_vel) if qj_vel is not None else list(z)
        msg.kp = list(z)
        msg.kd = list(z)
        msg.v_ee = list(v_ee) if v_ee is not None else [0.0] * 6
        msg.ee_frame = bool(ee_frame)
        with self._pub_lock:
            self._latest_cmd = msg
            try:
                self._lc.publish(CONTROL_CHANNEL, msg.encode())
            except Exception:
                pass

    def _republish_latest_cmd(self):
        # Re-stream (hold) the last command so the bridge keeps servoing the sim arm.
        # For a Cartesian jog this also keeps it running: the bridge re-solves the
        # velocity IK every sim step, so the same twist tracks the moving Jacobian.
        if self._lc is None:
            return
        with self._pub_lock:
            msg = self._latest_cmd
            if msg is None:
                return
            msg.timestamp = time.time_ns()
            try:
                self._lc.publish(CONTROL_CHANNEL, msg.encode())
            except Exception:
                pass

    def _publish_gripper(self, command):
        # The bridge holds the last jaw command, as the real jaws do, so it is sent once.
        if self._lc is None:
            return
        msg = ufactory_lite6_gripper_t()
        msg.timestamp = time.time_ns()
        msg.command = int(command)
        with self._pub_lock:
            try:
                self._lc.publish(GRIPPER_CHANNEL, msg.encode())
            except Exception:
                pass

    # --- Emulated controller queue ---

    def _flush_queue(self):
        with self._cmd_lock:
            self._epoch += 1

    def _enqueue(self, action, *args):
        item = _QueueItem(self._epoch, action, args)
        self._queue.put(item)
        return item

    def _queue_worker(self):
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            item.ok = item.epoch == self._epoch and item.action(item.epoch, *item.args)
            item.done.set()

    def _send_if_current(self, epoch, send):
        with self._cmd_lock:
            if epoch != self._epoch:
                return False
            send()
            return True

    def _run_gripper(self, epoch, command):
        return self._send_if_current(epoch, lambda: self._publish_gripper(command))

    def _run_move(self, epoch, q, speed, mvacc):
        q = np.asarray(q, dtype=float)
        if not self._send_if_current(epoch, lambda: self._send_control(
                qj_pos=q.tolist(), mode=MODE_POSITION, speed=speed, mvacc=mvacc)):
            return False
        # Done, or dropped by a flush (stop, jog, enable): STOP also ends a wait for a
        # sim that stopped answering.
        arrived_at = None
        while self._running and epoch == self._epoch:
            now = time.monotonic()
            state, world = self._latest()
            if world is not None and world.stopped:
                return False
            if (state is not None and world is not None and world.motion_done
                    and np.max(np.abs(np.asarray(world.q_des) - q)) < SETTLE_POS_RAD):
                if arrived_at is None:
                    arrived_at = now
                settled = (np.max(np.abs(np.asarray(state.qj_pos[:self.num_joints]) - q)) < SETTLE_POS_RAD
                           and np.max(np.abs(state.qj_vel[:self.num_joints])) < SETTLE_VEL_RAD_S)
                if settled or now - arrived_at > SETTLE_GIVE_UP_S:
                    return True
            time.sleep(0.005)
        return False

    def _within_limits(self, q):
        q = np.asarray(q, dtype=float)
        lo = np.array([lim[0] for lim in self.joint_limits]) - 1e-6
        hi = np.array([lim[1] for lim in self.joint_limits]) + 1e-6
        return q.shape == (self.num_joints,) and np.all(np.isfinite(q)) and np.all((q >= lo) & (q <= hi))

    def _move_and_wait(self, q, speed):
        # Queue a move without the initialized check (for initialize and sleep) and wait for it.
        item = self._enqueue(self._run_move, list(q), speed, self.mvacc_rad_s2)
        item.done.wait()
        return item.ok

    def _wait_for(self, ready, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if ready():
                return True
            time.sleep(0.01)
        return False

    # --- Viewer overlays (MuJoCo only - these never move the arm) ---

    def set_ghost(self, joint_angles_rad):
        # Show a translucent preview arm at these joint angles (rad); None hides it.
        # Published immediately so the viewer reacts to a button press without
        # waiting for the next SimArmThread tick.
        self._ghost_q = list(joint_angles_rad) if joint_angles_rad is not None else None
        self.publish_display()

    def set_fk_display(self, on):
        # Show or hide the FK pose marker (translucent box + axis triad).
        self._show_fk = bool(on)
        self.publish_display()

    def publish_display(self, ee_pose=None):
        # Publish the viewer overlay state. ee_pose is [x, y, z (mm), roll, pitch,
        # yaw (rad)] from the student FK; the wire format is SI, so mm -> m here.
        if self._lc is None:
            return
        if ee_pose is None:
            ee_pose = self.get_ee_pose()
        msg = ufactory_lite6_display_t()
        msg.timestamp = time.time_ns()
        msg.show_ghost = self._ghost_q is not None
        msg.ghost_qj_pos = list(self._ghost_q) if self._ghost_q is not None else [0.0] * self.num_joints
        msg.show_fk = self._show_fk
        msg.fk_pos = [v / 1000.0 for v in ee_pose[:3]]
        msg.fk_rpy = list(ee_pose[3:6])
        with self._pub_lock:
            try:
                self._lc.publish(DISPLAY_CHANNEL, msg.encode())
            except Exception:
                pass

    # --- Arm lifecycle ---

    def initialize(self):
        # Clear any stop or fault and drive to the initial pose, blocking like the
        # real arm's set_servo_angle(wait=True). False if the sim does not answer or
        # the move fails.
        if not self.connected:
            return False
        self.initialized = False
        if not self._wait_for(lambda: self.get_joint_angles() is not None, 2.0):
            print("WARNING: no state from the sim. Is mujoco-sim running?")
            return False
        self.enable()
        initial_pose = Q_DEFAULT  # matches UFactory Studio default, also the IK_numerical seed
        if not self._move_and_wait(initial_pose, INIT_SPEED_RAD_S):
            return False
        self.initialized = True
        return True

    def sleep(self):
        # Like the real arm: clear any stop, drive to the zero pose, then no motion
        # until initialize().
        if self.connected and self.get_joint_angles() is not None:
            self.enable()
            self._move_and_wait([0.0] * self.num_joints, INIT_SPEED_RAD_S)
        self.initialized = False

    # --- Motion ---

    def set_joint_angles(self, joint_angles_rad, wait=False):
        # Queue a speed/accel-limited move to these 6 joint angles (rad). Moves run
        # one after another, as on the real arm. wait=True blocks until this move is
        # done. Returns False if the move was rejected or, with wait, did not finish.
        if not (self.connected and self.initialized) or self._stopped():
            return False
        if not self._within_limits(joint_angles_rad):
            print(f"WARNING: set_joint_angles rejected, outside joint limits: {np.round(joint_angles_rad, 3)}")
            return False
        # Speed is fixed when the move is queued, as on the real arm.
        speed = max(self._speed_rad_s(), MIN_MOVE_SPEED_RAD_S)
        item = self._enqueue(self._run_move, [float(v) for v in joint_angles_rad], speed, self.mvacc_rad_s2)
        if not wait:
            return True
        item.done.wait()
        return item.ok

    # --- joint velocity (jog) mode: stream a velocity command the bridge integrates ---
    def enter_jog_mode(self):
        # Switch to joint-velocity mode; start held (zero velocity holds the current pose).
        self._flush_queue()
        self._send_control(qj_vel=[0.0] * self.num_joints, mode=MODE_JOINT_VEL,
                           mvacc=self.mvacc_rad_s2)

    def start_jog(self, joint_idx, direction):
        # Command one joint's velocity; the bridge integrates it while it keeps arriving.
        if not (self.connected and self.initialized) or self._stopped():
            return False
        self._flush_queue()
        jog_vel = [0.0] * self.num_joints
        jog_vel[joint_idx] = direction * self._speed_rad_s()
        self._send_control(qj_vel=jog_vel, mode=MODE_JOINT_VEL, mvacc=self.mvacc_rad_s2)
        return True

    def stop_jog(self):
        # Zero the jog velocity; the bridge ramps to a stop and holds. Stay in velocity mode.
        if not (self.connected and self.initialized):
            return True
        self._send_control(qj_vel=[0.0] * self.num_joints, mode=MODE_JOINT_VEL,
                           mvacc=self.mvacc_rad_s2)
        return True

    # --- Cartesian velocity (jog) mode: stream an EE twist the bridge runs velocity IK on ---
    def enter_cartesian_jog_mode(self):
        # Switch to Cartesian velocity mode; start held (a zero twist holds the current pose).
        self._flush_queue()
        self._send_control(v_ee=[0.0] * 6, mode=MODE_CART_VEL,
                           speed=self._speed_rad_s(), mvacc=self.mvacc_rad_s2)
        return True

    def start_cartesian_jog(self, axis_idx, direction, in_ee_frame=False):
        # Jog one Cartesian axis: 0-2 are linear (mm/s), 3-5 angular (rad/s), same
        # scaling as the real arm so the speed slider means the same thing on both.
        vel = [0.0] * 6
        scale = CART_JOG_LIN_MM_S if axis_idx < 3 else CART_JOG_ROT_RAD_S
        vel[axis_idx] = direction * self.speed_pct / 100.0 * scale
        return self.set_cartesian_velocity(vel, in_ee_frame)

    def set_cartesian_velocity(self, vel, in_ee_frame=False):
        # Send a full 6-vector Cartesian velocity: vel = [vx, vy, vz, wrx, wry, wrz]
        # (mm/s, rad/s), matching the real arm's units. The wire format is SI, so mm -> m here.
        if not (self.connected and self.initialized) or self._stopped():
            return False
        self._flush_queue()
        v_ee = [v / 1000.0 for v in vel[:3]] + list(vel[3:6])
        self._send_control(v_ee=v_ee, ee_frame=in_ee_frame, mode=MODE_CART_VEL,
                           speed=self._speed_rad_s(), mvacc=self.mvacc_rad_s2)
        return True

    def stop_cartesian_jog(self):
        # Zero the twist; the bridge ramps to a stop and holds. Stay in Cartesian mode.
        if not (self.connected and self.initialized):
            return
        self._send_control(v_ee=[0.0] * 6, mode=MODE_CART_VEL,
                           speed=self._speed_rad_s(), mvacc=self.mvacc_rad_s2)

    def has_fault(self):
        return False  # sim has no hardware fault state (yet)

    def pause(self):
        pass  # NOTE: Unimplemented

    def resume(self):
        pass  # NOTE: Unimplemented

    def stop(self):
        # Hard stop, like the real arm's stop state: flush the queue and latch a stop
        # in the bridge, which holds the arm and ignores motion from every client
        # until enable() or initialize().
        self._flush_queue()
        self._send_control(mode=MODE_STOP)
        self._wait_for(self._stopped, LATCH_TIMEOUT_S)

    def enable(self):
        # Clear a stop and hold the current pose (the real arm's set_state(0)).
        self._flush_queue()
        angles = self.get_joint_angles()
        if not self.connected or angles is None:
            return
        self._send_control(qj_pos=angles, mode=MODE_RESUME)
        # The bridge keeps only the newest command each step, so a command sent right
        # after could replace the resume. Wait until the stop latch is gone.
        self._wait_for(lambda: not self._stopped(), LATCH_TIMEOUT_S)

    def set_teach_mode(self, on):
        pass  # NOTE: Unimplemented

    # --- Gripper (binary jaws, same calls and semantics as Lite6Arm) ---

    def _command_gripper(self, command, wait, sync):
        if not self.connected:
            return
        if sync:
            # Runs after every move queued before it, like the xArm's queued tool IO.
            self._enqueue(self._run_gripper, command)
        else:
            self._publish_gripper(command)
        if wait:
            time.sleep(GRIPPER_DWELL_S)  # from the call, as on hardware

    def open_gripper(self, wait=False, sync=True):
        # Open the jaws. sync=False acts immediately instead of after queued moves.
        self._command_gripper(ufactory_lite6_gripper_t.OPEN, wait=wait, sync=sync)

    def close_gripper(self, wait=False, sync=True):
        # Close (grip). The jaws stall on a block with a capped squeeze.
        self._command_gripper(ufactory_lite6_gripper_t.CLOSE, wait=wait, sync=sync)

    def stop_gripper(self, sync=True):
        # Cut the jaw drive: the jaws stay where they are.
        self._command_gripper(ufactory_lite6_gripper_t.STOP, wait=False, sync=sync)

    # --- Kinematics / state accessors ---

    def get_joint_angles(self):
        # Current joint angles (rad) from the latest received sim state, or None.
        state = self._latest()[0]
        if state is None:
            return None
        return list(state.qj_pos[:self.num_joints])

    def get_ee_pose_sdk(self):
        return [0.0] * 6  # NOT Used in sim

    def get_ik_sdk(self, pose):
        return None  # NOT Used in sim

    def get_ee_pose(self):
        # Student FK: base -> EE pose (mm, rad) from the latest joint angles.
        if self.dh_params is None:
            return [0.0] * 6
        # T = FK_dh(self.dh_params, self.joint_angles, self.num_joints)  # DH method
        T = FK_pox(self.joint_angles, M, S_list)                         # PoX method
        if T is None:
            return [0.0] * 6
        return get_pose_from_T(T)

    def get_ik(self, pose):
        # Student IK: return joint angles (rad) for the requested pose (mm, rad), or None if no solution.
        if self.dh_params is None:
            return None
        return IK_geometric(self.dh_params, pose)

    def get_ik_numerical(self, pose):
        # Student IK, numerical: seeded from the default (home) pose, not the
        # current one, so the same target always yields the same elbow-up answer.
        if self.dh_params is None:
            return None
        return IK_numerical(self.dh_params, pose, q0=Q_DEFAULT,
                            joint_limits=self.joint_limits)

    def stop_link(self):
        # Tear down the LCM and queue threads. Anything still queued fails,
        # so no caller is left waiting.
        self._running = False
        self.connected = False
        self._flush_queue()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=0.5)
        while not self._queue.empty():
            self._queue.get_nowait().done.set()


class SimArmThread(QThread):
    """Polls sim joint state at 20 Hz; EE pose at 10 Hz.

    Re-streams the latest control target each tick so the sim bridge keeps holding.
    """

    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, arm, parent=None):
        QThread.__init__(self, parent=parent)
        self.arm = arm
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        tick = 0
        while self._running:
            if not self.arm.connected:
                time.sleep(0.25)
                continue

            # Re-stream (hold) the last command.
            self.arm._republish_latest_cmd()

            angles = self.arm.get_joint_angles()
            if angles is not None:
                self.arm.joint_angles = angles
                self.updateJointReadout.emit(list(self.arm.joint_angles))
                if tick % 2 == 0:
                    ee_pose = self.arm.get_ee_pose()
                    self.updateEndEffectorReadout.emit(ee_pose)
                    # Same FK result drives the viewer's FK marker. Re-publishing
                    # every tick also lets a late-starting bridge catch up.
                    self.arm.publish_display(ee_pose)
            tick += 1
            time.sleep(0.05)


if __name__ == "__main__":
    arm = SimArm()
    print(f"SimArm connected={arm.connected}")
    print("initialize:", arm.initialize())
    for _ in range(20):
        print("joint angles:", arm.get_joint_angles())
        time.sleep(0.1)

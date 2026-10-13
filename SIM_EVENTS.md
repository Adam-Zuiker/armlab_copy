# Simulation: gripper, blocks and Event 0

The MuJoCo sim now has the Arm Lab gripper, the board and the twelve blocks, so
you can write and test pick-and-place and competition code before your turn on
the real arm. Block poses come straight from the sim, so no camera work is
needed here. The code you write runs unchanged on the real arm.

## Get the update

```bash
git pull upstream main
```

Use plain `git pull` if you cloned the course repository directly. Nothing has
to be reinstalled: `mujoco-sim` runs from your checkout.

This update also changes four files you may have edited: `src/lite6arm.py`,
`src/control_station.py`, `src/state_machine.py` and `src/ui/layout.py`. If git
reports a conflict in one of them, keep your lines and the update's lines. In
`lite6arm.py`, take the update's `set_joint_angles` (it now has `wait=` and
returns True or False) and keep your `XARM_IP`. Then `git add` the file and
`git commit`.

## Install it on your own computer

On Linux (x86_64) or a Mac with Apple Silicon, with conda installed:

```bash
bash install_scripts/install_sim_env.sh
```

It creates the conda env `env550lab` with only what the sim and the control
station need. A Mac gets no RealSense support and starts the sim with
`mjpython -m lite6_sim.main` instead of `mujoco-sim`. If LCM then reports "No
route to host", run `sudo route add -nethost 239.255.76.67 -interface lo0` (it
lasts until the next reboot).

## Run it

Terminal A, the sim (on a Mac: `mjpython -m lite6_sim.main`):

```bash
mujoco-sim
```

Terminal B, the control station:

```bash
cd src
python control_station.py --sim
```

The Custom Buttons row along the bottom runs events through the state machine:

| Button | What it does |
| --- | --- |
| Reset Board (Sim) | drives the arm to the initial pose and lays the blocks out for Event 0 |
| Event 0 | one pick and place (`event_0` in `src/competition.py`) |

The other buttons are yours to bind. STOP stops the arm and so ends a running
event. Click Initial Pose to recover.

## What is simulated

- **Frame**: the world frame is the arm base frame. The origin is the centre of
  the arm mount, x points forward over the board, y to the arm's left, z up, and
  z = 0 is the board surface. Units are mm and rad.
- **Board**: 100 by 60 cm, x from -100 to 500 mm, y from -500 to 500 mm, 50 mm
  grid.
- **Gripper**: binary jaws, like the real one. `arm.open_gripper()`,
  `arm.close_gripper()`, `arm.stop_gripper()`. The jaws open to 50 mm and close
  on a block with a capped squeeze. The grasp point (middle of the finger pads)
  is 85 mm from the flange and the finger tips are 96.4 mm from it.
- **Blocks**: 38 mm and 25 mm cubes, one of each size in red, orange, yellow,
  green, blue and violet. Blocks an event does not use wait beside the board.
- **Perception**: `camera.detect_blocks()` returns a `list[Block]` (see
  `src/blocks.py`) with the exact pose of every block on the board that a
  camera looking straight down would see, i.e. no other block is above its
  centre. After you take a block off a stack, call it again to see the one
  underneath.

## The sim behaves like the real arm

- `arm.set_joint_angles(q)` queues the move and returns at once. Moves run one
  after another. `arm.set_joint_angles(q, wait=True)` blocks until the move is
  done and returns False if it failed or the arm was stopped.
- Gripper calls queue behind moves (`sync=True`, the default), so `close_gripper()`
  right after a queued move closes the jaws when the arm gets there.
  `sync=False` acts at once. `wait=True` sleeps 0.8 s from the call, so it does
  not wait for queued moves. Gripper calls return nothing.
- STOP stops the arm. Motion calls return False until you click Initial Pose,
  so check return values and give up early.

## Write your events

`src/competition.py` is yours. It has:

- `event_0(arm, camera)`: the warm-up event, a working pick and place. Read it
  first.
- Helpers: `top_down_pose()`, `jaw_yaw()`, `move_to()`, `grasp_height()`,
  `pick(arm, block)` and `place(arm, block, x, y, z_surface, yaw)`.

To put an event on a button, write it in `src/competition.py` as a function
`my_event(arm, camera)` and add `("My event", competition.my_event)` to
`CUSTOM_BUTTONS` in `src/control_station.py`. A click has the state machine run
it in its own thread.

The helpers use `arm.get_ik()`, i.e. your `IK_geometric` from `kinematics.py`,
and fall back to `IK_numerical` when it returns None.

On the real arm the same buttons run the same code. Implement
`Camera.detect_blocks()` in `src/camera.py` (Checkpoint 3) so it returns the same
`list[Block]`: block centres in the world frame in mm, the edge length, the colour
name and the yaw.

## Event 0 (sim warm-up)

- **Start**: all twelve blocks lie flat in front of the arm (x > 0), each at its
  own position and yaw. The layout is the same every time.
- **Each click** of Event 0 picks one random block in front of the arm and places
  it just behind the arm (x = -25 mm), on the lowest of six slots at y = -400,
  -300, -200, 200, 300 and 400 mm (the first one on a tie), then goes back to the
  initial pose. The first six clicks fill the slots in that order, the next six
  stack a second layer on top.
- Reset Board (Sim) lays the blocks out again. There are no points.

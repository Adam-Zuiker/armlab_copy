# Arm Lab - UFactory Lite 6

A classroom robotics arm lab using the **UFactory Lite 6** (6-DOF) and **Intel RealSense L515**.  

## Setup

1. Clone this repo:
    ```bash
    git clone https://gitlab.eecs.umich.edu/rob550-f26/armlab-f26.git
    cd armlab-f26
    ```

    > **Note:** The MuJoCo simulation bridge lives in `src/mujoco_sim/`, derived from
    > [`arc-bridge`](https://github.com/ARCaD-Lab-UM/arc-bridge) and trimmed to the Lite 6.
    > The generated LCM bindings are git-tracked, so no `lcm-gen` / `javac` step is needed.

2. Install Anaconda:
    ```bash
    ./install_scripts/install_anaconda.sh
    source ~/.bashrc
    ```

3. Create the `env550lab` environment and install the simulation bridge into it:
    ```bash
    ./install_scripts/install_conda_env.sh
    ```
    This provides the `mujoco-sim` and `550-lcm-spy` commands.

4. Install the camera stack and system packages (builds librealsense from source - 5~20 min):
    ```bash
    ./install_scripts/install_camera.sh
    source ~/.bashrc
    ```
    This step must come after step 3: the RealSense Python binding is built against the
    `env550lab` interpreter, so the environment has to exist first. It also appends
    `conda activate env550lab` to `~/.bashrc`, so every terminal you open after this
    starts in the environment already.

See `install_scripts/README.md` for what each script does and why, and
`src/mujoco_sim/README.md` for details on the bridge itself.

> **Everything runs in `env550lab`.** PyQt5, OpenCV, the xArm SDK and the AprilTag detector are all
> pinned in `src/mujoco_sim/environment_py310.yml`. If `conda activate env550lab` is missing, imports fail.

> **USB 3.2 required.** The L515 only exposes its full resolution (1280×720 color, 1024×768 depth)
> over USB 3.2 Gen 1 or better. On USB 2, only 640×480 is available.

## How to run

```bash
conda activate env550lab
cd src
python control_station.py            # --real (default), talks to the arm hardware
```

To run against the MuJoCo simulation instead, start the bridge in a **separate terminal**
first, then launch the GUI with `--sim`:

```bash
conda activate env550lab && mujoco-sim    # terminal A
conda activate env550lab && cd src && python control_station.py --sim   # terminal B
```

To test the camera independently:

```bash
conda activate env550lab && cd src
python camera.py      # opens RGB, Depth, and Tags windows
```

## Utilities

### `utils/camera_calibration.py` - measure the camera intrinsics

`camera.py` takes its intrinsics straight from the RealSense at startup. This tool lets you
measure them yourself with a checkerboard and compare the two, which is the point of the
calibration exercise - it does **not** feed the control station.

```bash
conda activate env550lab
cd utils
python camera_calibration.py
```

Before the first run, set the three constants at the top of the file to match your printed
board. They are **inner corners**, not squares - a 10x7-square board is 9x6:

```python
BOARD_COLS = 9        # inner corners across
BOARD_ROWS = 6        # inner corners down
SQUARE_MM  = 25.0     # measure one printed square with calipers
```

How to use it:

| Control | Does |
| --- | --- |
| **SPACE** | Capture the current frame |
| **Undo last** | Drop the most recent capture |
| **Calibrate** | Run the calibration - enabled at 10 frames |
| **Q** / **Quit** | Close |

Frames where the whole board is not visible are rejected, so everything you keep is usable.
Accepted frames leave a faint outline on the live view - use it to spread captures over the
frame, especially near the edges, which is what constrains the distortion terms. Tilt the
board 20-45 degrees between shots; head-on views alone cannot separate focal length from
distance.

Clicking **Calibrate** prints the camera's factory intrinsics next to the computed ones and
saves the result to `utils/camera_intrinsics.npz`:

```python
data = np.load("camera_intrinsics.npz")
K, dist = data["K"], data["dist"]
```

Expect an RMS reprojection error of roughly 0.1-0.5 px. Above 1.0 px, the usual cause is
`BOARD_COLS`/`BOARD_ROWS` counting squares instead of inner corners, or a `SQUARE_MM` that
does not match the printed board.

Note it uses **PyQt5** for display, not `cv2.imshow` - the env pins `opencv-python-headless`,
which has no GUI at all. See the gotcha below.

## Known gotchas

### Do NOT install `opencv-python` - use `opencv-python-headless`
`opencv-python` bundles its own Qt5 libraries which conflict with PyQt5 in the same process.
`opencv-python-headless` is identical except it has no bundled Qt - use this one.
`environment_py310.yml` already pins it correctly.

### `pyrealsense2` comes from a source build, not from the env
PyPI does carry `pyrealsense2==2.54.2.5684` with a matching `cp310` wheel, but this lab builds
2.54.2 from source in `install_camera.sh` and installs it to `/usr/local/OFF/`. The `PYTHONPATH`
entry in `~/.bashrc` makes it importable, including from inside `env550lab`.
If you open a new terminal and `import pyrealsense2` fails, run `source ~/.bashrc`.
If it still fails, the binding was probably built for a different interpreter - it is built for
one specific Python major.minor version. Check that the `.so` in `/usr/local/OFF/` is tagged
`cpython-310` to match `env550lab`:

```bash
ls /usr/local/OFF/            # want pyrealsense2.cpython-310-*.so
```

A `cpython-312` file means `install_camera.sh` was run against the system Python (Ubuntu 24.04)
instead of the env - re-run it with `env550lab` already created.

### AprilTag family
The lab uses **tagStandard41h12** tags. The family is set in `camera.py`:
```python
self.tag_detector = Detector(families='tagStandard41h12')
```
No external config file is needed.


### Arm IP address
The arm's IP is hardcoded in `src/lite6arm.py`. Change `XARM_IP` to match your station.


## Part 2.8c

To collect the data:

Press Calibrate while all four board tags are visible, then grip the fixture.
Jog to each of the 4 board locations at z ≈ 50, 150 and 250 mm.
At each pose, let the arm settle and press Ctrl+R.
Each press prints the error to the terminal and shows it in the status bar. It also rewrites check_3d_results.csv in whatever directory you launched from (probably src/).

FIXTURE_OFFSET_MM is still [0, 0, 0]. Fill it in from the fixture spec before you record, or every error will include the full fixture offset.

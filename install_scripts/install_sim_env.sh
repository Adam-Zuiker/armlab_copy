#!/usr/bin/env bash
# Minimal setup for the Arm Lab simulator on your own computer: creates the conda env
# env550lab with only what mujoco-sim and the control station (--sim) need. Needs
# conda already installed. Safe to run again.
#
# Supported: Linux x86_64, and macOS on Apple Silicon (no RealSense camera there).
#
#   bash install_scripts/install_sim_env.sh
set -eo pipefail   # no -u: conda's activate scripts read unset variables

ENV_NAME="${ENV_NAME:-env550lab}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Same versions as the lab machines (src/mujoco_sim/environment_py310.yml).
PACKAGES=(
    numpy==2.0.1
    scipy==1.15.3
    mujoco==3.3.7
    lcm==1.5.1
    pyqt5==5.15.11
    opencv-python-headless==4.10.0.84
    pupil-apriltags==1.0.4.post11
    xarm-python-sdk==1.18.4
)

case "$(uname -s)/$(uname -m)" in
    Linux/x86_64)
        # 2.54.2 is the last release that supports the L515. It needs Python 3.10.
        PACKAGES+=(pyrealsense2==2.54.2.5684)
        SIM_CMD="mujoco-sim" ;;
    Darwin/arm64)
        # No pyrealsense2 for macOS: the control station runs without video.
        # MuJoCo's viewer only runs under its mjpython launcher on macOS.
        SIM_CMD="mjpython -m lite6_sim.main" ;;
    Darwin/x86_64)
        echo "Intel Macs are not supported: lcm 1.5.1 has no build for them." >&2
        exit 1 ;;
    *)
        echo "Unsupported platform $(uname -s)/$(uname -m). Use Linux x86_64 or a Mac with Apple Silicon." >&2
        exit 1 ;;
esac

if ! command -v conda > /dev/null 2>&1; then
    echo "conda not found. Install Miniconda or Anaconda, open a new terminal and run this again." >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# conda-forge only: the Anaconda channels make new conda versions ask to accept their terms.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Using the existing env $ENV_NAME"
else
    echo "==> Creating the env $ENV_NAME (Python 3.10)"
    conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.10 pip
fi
conda activate "$ENV_NAME"

PY_VERSION="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$PY_VERSION" != "3.10" ]; then
    echo "The env $ENV_NAME has Python $PY_VERSION, but this setup needs 3.10." >&2
    echo "Remove it (conda env remove -n $ENV_NAME) or set ENV_NAME to another name, then run this again." >&2
    exit 1
fi

# Packages in ~/.local can shadow the env's own (an old opencv there breaks Qt).
conda env config vars set -n "$ENV_NAME" PYTHONNOUSERSITE=1 > /dev/null
export PYTHONNOUSERSITE=1

echo "==> Installing Python packages"
python -m pip install "${PACKAGES[@]}"
echo "==> Installing the simulator from this checkout"
python -m pip install --no-deps -e "$REPO_DIR/src/mujoco_sim"

echo "==> Checking the install"
python - "$REPO_DIR" << 'EOF'
import sys
import cv2, lcm, mujoco, numpy, pupil_apriltags, scipy, xarm
from PyQt5 import QtWidgets
from lite6_sim.lcm_msgs import sim_world_t
model = mujoco.MjModel.from_xml_path(sys.argv[1] + "/src/mujoco_sim/robot_assets/UfactoryLite6/xml/armlab_scene.xml")
try:
    import pyrealsense2
    camera = "RealSense support"
except ImportError:
    camera = "no RealSense support"
print(f"OK: MuJoCo {mujoco.__version__}, sim model loaded ({model.nbody} bodies), {camera}")
EOF

cat << EOF

Done. Run the sim and the control station in two terminals:

    conda activate $ENV_NAME
    $SIM_CMD

    conda activate $ENV_NAME
    cd "$REPO_DIR/src" && python control_station.py --sim
EOF

#!/usr/bin/env bash
# Run the G1 controller in mc_mujoco. Extra arguments go to mc_mujoco, e.g. --without-visualization.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# mc_rtc resolves the package:// URIs of the URDF through the ament index.
export AMENT_PREFIX_PATH="${G1_INSTALL_PREFIX:-$HOME/workspace/install}:${AMENT_PREFIX_PATH:-}"

# --torque-control is required: without it mc_mujoco ignores the torques and applies its own PD.
exec mc_mujoco -f "$root/etc/mc_rtc_g1.yaml" --torque-control "$@"

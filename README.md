# Physical AI Tools

This repository offers an interface for developing physical AI applications using LeRobot and ROS 2. For detailed usage instructions, please refer to the documentation below.
  - [Documentation for AI Worker](https://ai.robotis.com/)

To clone this repository:
```bash
git clone -b jazzy https://github.com/ROBOTIS-GIT/physical_ai_tools.git --recursive
```

To learn more about the ROS 2 packages for the AI Worker, visit:
  - [AI Worker ROS 2 Packages](https://github.com/ROBOTIS-GIT/ai_worker)

To explore our open-source platforms in a simulation environment, visit:
  - [Simulation Models](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)

For usage instructions and demonstrations of the AI Worker, check out:
  - [Tutorial Videos](https://www.youtube.com/@ROBOTISOpenSourceTeam)

To access datasets and pre-trained models for our open-source platforms, see:
  - [AI Models & Datasets](https://huggingface.co/ROBOTIS)

To use the Docker image for running ROS packages and Physical AI tools with the AI Worker, visit:
  - [Docker Images](https://hub.docker.com/r/robotis/ros/tags)

## ACT LiPo Post-Optimization

LiPo post-optimization only affects ACT policy inference. Other policies, such as Diffusion, pi0, pi0.5, and VLA, keep their original behavior.

Start the Physical AI server first:

```bash
ros2 launch physical_ai_server physical_ai_server_bringup.launch.py
```

Check or enable LiPo before starting ACT inference:

```bash
ros2 param get /physical_ai_server use_lipo
ros2 param set /physical_ai_server use_lipo true
```

Disable LiPo and restore the original ACT inference behavior:

```bash
ros2 param set /physical_ai_server use_lipo false
```

The parameter is read when inference starts, so set `use_lipo` before pressing Start Inference in the UI. If `use_lipo` is true but the selected policy is not ACT, LiPo is skipped.

The LeRobot submodule is not modified in this repository. Docker builds install LeRobot first, then apply the ACT-only LiPo hook to the installed LeRobot source with `physical_ai_server/scripts/apply_lerobot_act_lipo_patch.py`.

## LiPo Action Debug Plot

For debugging or result analysis, the ACT inference path can save both the raw policy action chunks and the LiPo-optimized action chunks as `.npy` files. These files can be plotted directly or reused with the `lab-dream/lipo` visualization notebook.

Enable debug action saving before starting ACT inference:

```bash
ros2 param set /physical_ai_server save_lipo_debug_actions true
ros2 param set /physical_ai_server lipo_debug_save_dir /workspace/lipo_debug_actions
ros2 param set /physical_ai_server lipo_debug_save_every_n 1
```

Then start ACT inference from the Web UI or by calling `/task/command`. During inference, the server writes the following files:

```text
/workspace/lipo_debug_actions/raw_action_chunks.npy
/workspace/lipo_debug_actions/optimized_action_chunks.npy
/workspace/lipo_debug_actions/log_inference_actions_raw.npy
/workspace/lipo_debug_actions/log_inference_actions_optimized.npy
/workspace/lipo_debug_actions/log_inference_actions.npy
/workspace/lipo_debug_actions/metadata.json
```

The file shapes are:

```text
raw_action_chunks.npy              (num_chunks, chunk_size, action_dim)
optimized_action_chunks.npy        (num_chunks, chunk_size, action_dim)
log_inference_actions_raw.npy      (num_chunks * chunk_size, action_dim)
log_inference_actions_optimized.npy (num_chunks * chunk_size, action_dim)
log_inference_actions.npy          same as raw flat data, for lipo_visualization.ipynb compatibility
```

When using the Docker compose setup in this repository, `/workspace` is mounted to `physical_ai_tools/docker/workspace` on the host. The default host-side output path is therefore:

```text
physical_ai_tools/docker/workspace/lipo_debug_actions
```

Create a quick raw-vs-optimized plot on the host:

```bash
cd /home/son/git_clone_project/ffw/physical_ai_tools/docker/workspace/lipo_debug_actions

python3 - <<'PY'
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

base = Path(".")
metadata = json.loads((base / "metadata.json").read_text())
raw = np.load(base / "log_inference_actions_raw.npy")
optimized = np.load(base / "log_inference_actions_optimized.npy")

fps = metadata.get("fps") or 30.0
joint_index = 0
t = np.arange(raw.shape[0]) / fps

plt.figure(figsize=(12, 4))
plt.plot(t, raw[:, joint_index], label="Raw action", linewidth=1.0)
plt.plot(t, optimized[:, joint_index], label="LiPo optimized action", linewidth=1.4)
plt.xlabel("Time (s)")
plt.ylabel(f"Action dim {joint_index}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("action_plot_joint0.png", dpi=200)
print("Saved action_plot_joint0.png")
PY
```

To use the `lab-dream/lipo` notebook, copy either the raw or optimized flat action file to the notebook's expected data path:

```bash
git clone https://github.com/lab-dream/lipo.git
cd lipo

# Raw ACT chunks: use this when you want the notebook to run ActionLiPo again.
cp /home/son/git_clone_project/ffw/physical_ai_tools/docker/workspace/lipo_debug_actions/log_inference_actions_raw.npy \
  data/log_inference_actions.npy

# Or inspect the already optimized chunks instead.
# cp /home/son/git_clone_project/ffw/physical_ai_tools/docker/workspace/lipo_debug_actions/log_inference_actions_optimized.npy \
#   data/log_inference_actions.npy
```

In `lipo_visualization.ipynb`, set the visualization parameters from `metadata.json`. For example, with the current ACT model:

```python
chunk = 100
blend = 10
time_delay = 5
dt = 1.0 / 30.0
```

If a different ACT model is used, read `chunk_size`, `blending_horizon`, `len_time_delay`, and `fps` from `metadata.json` instead of hard-coding them.

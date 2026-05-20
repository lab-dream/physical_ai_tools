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

LiPo post-optimization is disabled by default and only affects ACT policy inference. Other policies, such as Diffusion, pi0, pi0.5, and VLA, keep their original behavior.

Start the Physical AI server first:

```bash
ros2 launch physical_ai_server physical_ai_server_bringup.launch.py
```

Enable LiPo before starting ACT inference:

```bash
ros2 param set /physical_ai_server use_lipo true
```

Disable LiPo and restore the original ACT inference behavior:

```bash
ros2 param set /physical_ai_server use_lipo false
```

The parameter is read when inference starts, so set `use_lipo` before pressing Start Inference in the UI. If `use_lipo` is true but the selected policy is not ACT, LiPo is skipped.

The LeRobot submodule is not modified in this repository. Docker builds install LeRobot first, then apply the ACT-only LiPo hook to the installed LeRobot source with `physical_ai_server/scripts/apply_lerobot_act_lipo_patch.py`.

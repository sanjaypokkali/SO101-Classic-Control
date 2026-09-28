# SO101 SDK
The Lerobot SO101 arm does not have a traditional SDK like most robotic arms. This repository is meant to provide a SDK-like interface for the SO101 arm.

Currently has support for:

-> Gripper Control

-> Joint Position Control

-> Cartesian Position Control

## Install

```bash
pip install -e .
```

Or you can just add this repo to your project and use it as is!

## Usage

Before first use (or if motors feel off), run LeRobot's calibration and copy over the calibration file to so101_control/calibration/so101_arm.json

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0 --robot.id=so101_arm
```

```python
from so101_control import SO101Interface

arm = SO101Interface(port="/dev/ttyUSB0")
arm.connect()

arm.set_joint_positions([0.0, 0.0, 0.0, 0.0, 0.0])
arm.set_eef_pose([0.3, 0.0, 0.2, 0.0, 0.0, 0.0])  # x, y, z, roll, pitch, yaw
arm.set_gripper_position(0.05)
```

If commanding the arm to `[0, 0, 0, 0, 0]` doesn't land it exactly at the zero pose, measure the error per joint (in degrees) and add it to `self.joint_offsets_deg` in [so101_interface.py](so101_control/so101_interface.py) to correct the calibration.

## Tests

```bash
python3 -m unittest discover -s tests -p "test_so101_interface.py" -v
```

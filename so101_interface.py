import logging
import json
import time
import numpy as np
import pinocchio as pin
from kinematics import Kinematics
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

class SO101Interface():
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        urdf_path: str = "./urdf/so101_new_calib.urdf",
        ee_link_name: str = "gripper_frame_link",
        calibration_path: str = "./calibration/so101_arm.json",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.port = port
        self.urdf_path = urdf_path
        self.ee_link_name = ee_link_name
        self.calibration_path = calibration_path

        self.bus: FeetechMotorsBus | None = None

        self._connected = False
        self._enabled = False

        self.joint_offsets_deg = np.array(
            [0, 0.26373626, 2.59340659, 0.65934066, 0.21978022], dtype=float
        )

        # Motor configuration: 5 DOF arm + 1 gripper
        self.motor_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        self.gripper_name = "gripper"
        self.motor_ids: dict[str, int] = {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }
        self.gripper_max_open_m = 0.1
        self._control_dt = 0.02

        self.kinematics = Kinematics()

    # ============= Connection Management =============
    def _load_calibration(self, calibration_path: str = "") -> dict[str, MotorCalibration]:
        """Load motor calibration from JSON file."""
        try:
            with open(calibration_path) as f:
                calib_data = json.load(f)
            calibration: dict[str, MotorCalibration] = {}
            for name, data in calib_data.items():
                calibration[name] = MotorCalibration(**data)
            return calibration
        except Exception as e:
            self.logger.warning(f"Failed to load calibration from {calibration_path}: {e}")
            return {}

    def configure_motors(self) -> None:
        """
        Configure motors: operating mode + PID.

        - Puts all motors (joints + gripper) into POSITION mode.
        - Sets PID coefficients tuned to reduce shakiness.
        """
        if not self.bus:
            raise RuntimeError("Bus not connected.")

        self.logger.info("Configuring motors (OperatingMode + PID gains)")

        # Disable torque while tweaking settings
        with self.bus.torque_disabled():
            # Let LeRobot configure registers (limits, accel, etc.)
            self.bus.configure_motors()

            for motor in self.bus.motors:
                # Position control for all motors
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

                # PID gains – tuned for smoother motion
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

        self.logger.info("Motor configuration complete")

    def connect(self) -> bool:
        """Connect to SO101 via bus.

        Args:
            config: Configuration with 'port' (e.g., 'dev/ttyUSB0') and config path

        Returns:
            True if connection successful
        """
        try:
            self.logger.info("Connecting to SO-101 arm on port %s", self.port)

            norm_mode_body = MotorNormMode.DEGREES
            motors: dict[str, Motor] = {}

            # Arm joints: normalized in degrees
            for name in self.motor_names:
                motors[name] = Motor(self.motor_ids[name], "sts3215", norm_mode_body)

            # Gripper: normalized in [0, 100]
            motors[self.gripper_name] = Motor(
                self.motor_ids[self.gripper_name],
                "sts3215",
                MotorNormMode.RANGE_0_100,
            )

            calibration = self._load_calibration(self.calibration_path)

            self.bus = FeetechMotorsBus(
                port=self.port,
                motors=motors,
                calibration=calibration,
            )
            self.bus.connect()
            self.logger.info("FeetechMotorsBus connected")

            # Configure modes, PID, etc.
            self.configure_motors()
            self._connected = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to SO-101 arm: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the arm."""
        if self.bus:
            self.bus.disconnect()
            self.logger.info("SO-101 bus disconnected")
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ============= Servo Control =============
    def enable_servos(self) -> bool:
        """Enable motor control.

        Returns:
            True if servos enabled
        """
        if not self.bus:
            return False
        self.bus.enable_torque()
        self._enabled = True
        return True


    def disable_servos(self) -> bool:
        """Disable motor control.

        Returns:
            True if servos disabled
        """
        if not self.bus:
            return False
        self.bus.disable_torque()
        self._enabled = False
        return True

    def are_servos_enabled(self) -> bool:
        """Check if servos are enabled.

        Returns:
            True if enabled
        """
        return self._enabled

    # ============= Joint Info Methods =============
    def get_joint_positions(self, degree: bool = False) -> list[float]:
        """
        Get current joint angles with joint offsets applied.

        Returns:
            Joint positions in RADIANS or DEGREES (default: RADIANS)
        """
        if not self.bus:
            return [0.0] * len(self.motor_names)

        values = self.bus.sync_read("Present_Position")
        q_deg_raw = np.array([values[name] for name in self.motor_names], dtype=float)
        q_deg = q_deg_raw - self.joint_offsets_deg
        return q_deg.tolist() if degree else np.radians(q_deg).tolist() 

    def get_joint_velocities(self) -> list[float]:
        """Get current joint velocities.

        Returns:
            Joint velocities in RAD/S (0-indexed)
        """
        try:
            if not self.bus:
                return [0.0] * len(self.motor_names)

            values = self.bus.sync_read("Present_Velocity")
            vel_deg = np.array([values[name] for name in self.motor_names], dtype=float)
            return np.radians(vel_deg).tolist()
        except Exception as e:
            self.logger.debug(f"Error when reading Present_Velocity: {e}")
            return [0.0] * len(self.motor_names)

    def move_ptp(self, positions: list[float], duration: float = 2.0) -> bool:
        q_target = np.asarray(positions, dtype=float)
        q_curr = np.array(self.get_joint_positions())  
        q_diff = q_target - q_curr

        max_diff = float(np.max(np.abs(q_diff)))
        if max_diff < 1e-6:
            q_deg = np.degrees(q_target)
            cmd = {name: float(q_deg[i]) for i, name in enumerate(self.motor_names)}
            self.bus.sync_write("Goal_Position", cmd)
            return True
        
        steps = max(int(duration / self._control_dt), 1)
        for i in range(1, steps + 1):
            alpha = i / steps
            q_interp = q_curr + alpha * q_diff
            q_deg = np.degrees(q_interp)
            cmd = {name: float(q_deg[j]) for j, name in enumerate(self.motor_names)}
            self.bus.sync_write("Goal_Position", cmd)
            time.sleep(self._control_dt)
        return False

    def set_joint_positions(self, positions: list[float], duration: float = 2.0) -> bool:
        """Move joints to target position.

        Args: Target positions in RADIANS

        Returns:
            True if successful, False if not
        """
        if not self.bus:
            return False

        return self.move_ptp(positions, duration)
    
    # ============= Cartesion Control Methods =============
    def get_eef_pose(self) -> np.ndarray:
        """
        Get end effector pose as a 4x4 homogeneous transform matrix.

        Args:
            None
        """
        eef_pose = self.kinematics.fk(self.get_joint_positions() + [0.0])
        return eef_pose.homogeneous

    def set_eef_pose(self, target_pose: list[float]) -> bool:
        """
        Move end effector to a target pose.

        Args:
            target_pose: [x, y, z, roll, pitch, yaw] in meters/radians
        """
        xyz = np.asarray(target_pose[:3], dtype=float)
        rpy = np.asarray(target_pose[3:], dtype=float)
        target_se3 = pin.SE3(pin.rpy.rpyToMatrix(rpy), xyz)

        curr_joint_angles = self.get_joint_positions() + [0.0]
        ik_joint_angles = self.kinematics.ik(curr_joint_angles, target_se3)
        arm_joint_angles = ik_joint_angles[: len(self.motor_names)]
        if self.set_joint_positions(arm_joint_angles):
            return True
        return False


    # ============= Gripper Methods =============
    def set_gripper_position(self, position: float) -> bool:
        """
        Move gripper to target position.

        Args:
            position: 0.0 (closed) to 0.1 (open) in meters.
        """
        if not self.bus:
            self.logger.warning("Robot not connected")
            return False

        # Map 0–0.1 m → 0–100 normalized range
        val = (position / 0.1) * 100.0
        val = max(0.0, min(100.0, val))
        self.bus.write("Goal_Position", self.gripper_name, val)
        return True

    def get_gripper_position(self) -> float | None:
        """
        Get gripper position.

        Returns:
            Position in meters ( 0.0 [closed] to 0.1 [open]) or None
        """
        if not self.bus:
            return None

        raw = float(self.bus.read("Present_Position", self.gripper_name))
        pos_m = (raw / 100.0) * self.gripper_max_open_m
        return max(0.0, min(self.gripper_max_open_m, pos_m))

    

    
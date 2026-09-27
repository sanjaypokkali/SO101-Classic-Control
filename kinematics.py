import numpy as np
import pinocchio as pin

class Kinematics():
    def __init__(self):
        self.urdf = "urdf/so101_new_calib.urdf"
        self.eef_link_name = "gripper_frame_link"
        self.model = pin.buildModelFromUrdf(self.urdf)
        self.data = self.model.createData()
        self.eef_id = self.model.getFrameId(self.eef_link_name)

    def fk(self, joint_angles: list[float]) -> pin.SE3:
        """
        Computes the forward kinematics to find the end-effector pose.

        Args:
            jointAngles: List of joint angles in radians.

        Returns:
            pin.SE3: The SE(3) transform of the end-effector frame.
                Access components via .translation, .rotation, or .homogeneous.
        """
        q = np.asarray(joint_angles, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.eef_id].copy()

    def ik(self, curr_joint_angles: list[float], target_pose: pin.SE3) -> list[float]:
        # IK parameters
        max_iters = 100
        eps = 1e-4
        damping = 1e-6
        alpha = 0.5   # step size
        new_joint_angles = curr_joint_angles
        for _ in range(max_iters):
            curr_cartesian_pose = self.fk(new_joint_angles)
            # Compute 6D error vector from current EEF pose to target pose.
            # We can't subtract SE3 matrices directly, so we:
            #   1. curr_pose.inverse() * target_pose → relative transform from curr to target
            #      (inverse "undoes" curr, bringing us to origin, then * target finds the path to target)
            #   2. pin.log(...)  → maps that SE3 matrix down to a flat 6D vector [vx,vy,vz, wx,wy,wz]
            #      (like flattening a globe to a map so we can do linear algebra on it)
            #   3. .vector      → extracts the numpy array from the log result
            curr_to_target_transform = curr_cartesian_pose.inverse() * target_pose 
            error = pin.log(curr_to_target_transform).vector

            if np.linalg.norm(error) < eps:
                print("Found a satisfactory IK solution!!")
                return list(new_joint_angles)
            
            # Jacobian columns represent the joints and the rows represent the eef. rows 1-3 correspond to linear velocity of xyz and 4-6 correspond to angular velocity of xyz
            J = pin.computeFrameJacobian(
                self.model, self.data, np.asarray(new_joint_angles, dtype=float), self.eef_id)

            # Damped least squares step: dq = J^T (J J^T + damping*I)^-1 * error
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, error)

            # Integrate on the joint manifold rather than a plain += so joint
            # types other than simple revolute (e.g. continuous/spherical) stay valid.
            new_joint_angles = pin.integrate(self.model, np.asarray(new_joint_angles, dtype=float), alpha * dq)

        print("Exceeded iterations before finding satisfactory solution")
        return list(new_joint_angles)

            

            
            
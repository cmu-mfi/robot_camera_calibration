import numpy as np
from scipy.spatial.transform import Rotation as R
from autolab_core import RigidTransform
from frankapy import FrankaArm
from scipy.spatial.transform import Rotation
import math

PI = np.pi
EPS = np.finfo(float).eps * 4.0

class FrankaRobot:
    def __init__(self):
        self.fa = FrankaArm()

    def get_eef_pose(self):
        '''
        Get the latest end effector pose using FrankaArm
        
        Returns: np.ndarray(4x4) transformation matrix
        '''
        # Get current end effector pose from FrankaArm
        return np.array(self.fa._state_client._get_current_robot_state().robot_state.O_T_EE).reshape(4, 4).transpose()
            
    def move_to_pose(self, position, orientation, max_velocity_scaling_factor=0.3, max_acceleration_scaling_factor=0.3):
        '''
        Move robot to specified pose using FrankaArm
        '''
        position = np.array(position)
        orientation = np.array(orientation)
        
        # Convert orientation if needed
        if orientation.shape == (3,3):
            rotation = R.from_matrix(orientation)
            orientation = rotation.as_quat()
        elif orientation.shape != (4,):
            return False

        # Create RigidTransform target pose
        target_rot = R.from_quat(orientation).as_matrix()
        target_pose = RigidTransform(rotation=target_rot,
                                   translation=position, 
                                   from_frame='franka_tool',
                                   to_frame='world')

        # Move to target pose
        self.fa.goto_pose(target_pose, duration=3)
        return True

    def reset_pose(self):
        '''Reset robot to home pose'''
        self.fa.reset_pose()

    def close_gripper(self):
        '''Close the gripper'''
        self.fa.close_gripper()

    def open_gripper(self):
        '''Open the gripper'''
        self.fa.open_gripper()

    def gripper_move_to(self, width):
        '''Move gripper to specified width'''
        if width == 0.08:
            self.fa.open_gripper()
        else:
            self.fa.goto_gripper(width=width)

    def move_by_delta(self, delta_pos=np.zeros(3), delta_rotation=np.zeros(3)):
        '''
        Move robot by delta position and rotation from current pose
        Args:
            delta_pos: [x,y,z] translation in meters
            delta_rotation: [rx,ry,rz] rotation in radians
        '''
        current_pose = self.get_eef_pose()
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]
        
        # Add delta position
        target_pos = current_pos + np.array(delta_pos)
        
        # Add delta rotation
        delta_rot_mat = R.from_euler('xyz', delta_rotation).as_matrix()
        target_rot = current_rot @ delta_rot_mat
        
        # Move to new pose
        target_quat = R.from_matrix(target_rot).as_quat()
        return self.move_to_pose(target_pos, target_quat)

    def get_joint_positions(self):
        '''Get current joint positions'''
        return self.fa.get_joints()

    def move_to_joints(self, joint_positions, duration=3):
        '''Move to specified joint positions'''
        self.fa.goto_joints(joint_positions, duration=duration)

    @property 
    def eef_axis_angle(self):
        current_rot = self.get_eef_pose()[:3, :3]
        quat = R.from_matrix(current_rot).as_quat()
        return self.__quat2axisangle(quat)

    @property
    def eef_pose(self):
        return self.get_eef_pose()

    @property 
    def eef_rot_and_pos(self):
        pose = self.fa.get_pose()
        return pose.rotation, pose.translation

    @property
    def joint_positions(self):
        return self.get_joint_positions()

    @property
    def gripper_position(self):
        return self.fa._state_client._get_current_robot_state().robot_state.O_T_EE
    
    def __quat2axisangle(self, quat):
        """
        Converts quaternion to axis-angle format.
        Returns a unit vector direction scaled by its angle in radians.

        Args:
            quat (np.array): (x,y,z,w) vec4 float angles

        Returns:
            np.array: (ax,ay,az) axis-angle exponential coordinates
        """
        # clip quaternion
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            # This is (close to) a zero degree rotation, immediately return
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def __quat2mat(self, quaternion):
        """
        Converts given quaternion to matrix.

        Args:
            quaternion (np.array): (x,y,z,w) vec4 float angles

        Returns:
            np.array: 3x3 rotation matrix
        """
        # awkward semantics for use with numba
        inds = np.array([3, 0, 1, 2])
        q = np.asarray(quaternion).copy().astype(np.float32)[inds]

        n = np.dot(q, q)
        if n < EPS:
            return np.identity(3)
        q *= math.sqrt(2.0 / n)
        q2 = np.outer(q, q)
        return np.array(
            [
                [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
                [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
                [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
            ]
        )

    def __mat2quat(self, rmat):
        """
        Converts given rotation matrix to quaternion.

        Args:
            rmat (np.array): 3x3 rotation matrix

        Returns:
            np.array: (x,y,z,w) float quaternion angles
        """
        M = np.asarray(rmat).astype(np.float32)[:3, :3]

        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
        if q1[0] < 0.0:
            np.negative(q1, q1)
        inds = np.array([1, 2, 3, 0])
        return q1[inds]

    def __quat_multiply(self, quaternion1, quaternion0):
        """
        Return multiplication of two quaternions (q1 * q0).

        E.g.:
        >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
        >>> np.allclose(q, [-44, -14, 48, 28])
        True

        Args:
            quaternion1 (np.array): (x,y,z,w) quaternion
            quaternion0 (np.array): (x,y,z,w) quaternion

        Returns:
            np.array: (x,y,z,w) multiplied quaternion
        """
        x0, y0, z0, w0 = quaternion0
        x1, y1, z1, w1 = quaternion1
        return np.array(
            (
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            ),
            dtype=np.float32,
        )

    def __quat_conjugate(self, quaternion):
        """
        Return conjugate of quaternion.

        E.g.:
        >>> q0 = random_quaternion()
        >>> q1 = quat_conjugate(q0)
        >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
        True

        Args:
            quaternion (np.array): (x,y,z,w) quaternion

        Returns:
            np.array: (x,y,z,w) quaternion conjugate
        """
        return np.array(
            (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
            dtype=np.float32,
        )

    def __quat_inverse(self, quaternion):
        """
        Return inverse of quaternion.

        E.g.:
        >>> q0 = random_quaternion()
        >>> q1 = quat_inverse(q0)
        >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
        True

        Args:
            quaternion (np.array): (x,y,z,w) quaternion

        Returns:
            np.array: (x,y,z,w) quaternion inverse
        """
        return self.__quat_conjugate(quaternion) / np.dot(quaternion, quaternion)
    
    def __quat_distance(self, quaternion1, quaternion0):
        """
        Returns distance between two quaternions, such that distance * quaternion0 = quaternion1

        Args:
            quaternion1 (np.array): (x,y,z,w) quaternion
            quaternion0 (np.array): (x,y,z,w) quaternion

        Returns:
            np.array: (x,y,z,w) quaternion distance
        """
        return self.__quat_multiply(quaternion1, self.__quat_inverse(quaternion0))
    
    def __estimate_tag_pose(self, finger_pose):
        """
        Estimate the tag pose given the gripper pose by applying the gripper-to-tag transformation.

        Args:
            finger_pose (eef_pose): 4x4 transformation matrix from gripper to robot base
        Returns:
            hand_pose: 4x4 transformation matrix from hand to robot base
            tag_pose: 4x4 transformation matrix from tag to robot base
        """
        from scipy.spatial.transform import Rotation

        # Estimate the hand pose
        # finger_to_hand obtained from the product manual: 
        # [https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf]
        finger_to_hand = np.array([
            [0.707,  0.707, 0, 0],
            [-0.707, 0.707, 0, 0],
            [0, 0, 1, 0.1034],
            [0, 0, 0, 1],
        ])
        finger_to_hand = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.1034],
            [0, 0, 0, 1],
        ])
        hand_to_finger = np.linalg.inv(finger_to_hand)
        print("hand to finger", hand_to_finger)
        hand_pose = np.dot(finger_pose, hand_to_finger)

        t_tag_to_hand = np.array([0.048914, 0.0275, 0.00753])
        # R_tag_to_hand = Rotation.from_quat([0.5, -0.5, 0.5, -0.5])
        R_tag_to_hand = Rotation.from_quat([0, 0, 0, 1])
        tag_to_hand = np.eye(4)
        tag_to_hand[:3, :3] = R_tag_to_hand.as_matrix()
        tag_to_hand[:3, 3] = t_tag_to_hand

        tag_pose = np.dot(hand_pose, tag_to_hand)
        
        return hand_pose, tag_pose

    def __compute_errors(self, pose_1, pose_2):
        pose_a = (
            pose_1[:3]
            + self.__quat2axisangle(np.array(pose_1[3:]).flatten()).tolist()
        )
        pose_b = (
            pose_2[:3]
            + self.__quat2axisangle(np.array(pose_2[3:]).flatten()).tolist()
        )
        return np.abs(np.array(pose_a) - np.array(pose_b))

    def __rpy2mat(self, rot_x, rot_y, rot_z):
        """Create a rotation matrix from Euler angles."""
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(rot_x), -np.sin(rot_x)],
            [0, np.sin(rot_x), np.cos(rot_x)]
        ])
        ry = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])
        rz = np.array([
            [np.cos(rot_z), -np.sin(rot_z), 0],
            [np.sin(rot_z), np.cos(rot_z), 0],
            [0, 0, 1]
        ])
        return rz @ ry @ rx
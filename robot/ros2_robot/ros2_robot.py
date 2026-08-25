import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.time import Time
from rclpy.action.client import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


class _PoseGoal:
    def __init__(
        self,
        frame_id=None,
        position=None,
        orientation=None,
        velocity=None,
        acceleration=None,
        method="PTP",
    ):
        if position is None or len(position) != 3:
            raise ValueError("Position should be 3 values (ex: [0.0, 0.0, 0.0])")
        if orientation is None or len(orientation) != 4:
            raise ValueError("orientation should be 4 values (ex: [0.0, 0.0, 0.0, 1.0])")
        if velocity is None:
            raise ValueError("Velocity needs to be specified!")
        if velocity is None:
            raise ValueError("Acceleration needs to be specified!")
        if frame_id is None:
            raise ValueError("frame_id needs to be specified!")
        self.position = position
        self.orientation = orientation
        self.frame_id = frame_id
        self.velocity = velocity
        self.acceleration = acceleration
        self.method = method


class Ros2Robot(Node):
    def __init__(self, robot_namespace="", base_frame="base", tool_frame="tool0"):
        if not rclpy.ok():
            rclpy.init()

        super().__init__("ros2_robot_client")
        self.ns = robot_namespace
        self.prefix = ""
        self.move_topic = "move_action"
        if self.ns != "":
            self.prefix = str(self.ns) + "_"
            self.move_topic = "/" + str(self.ns) + "/move_action"
        else:
            self.prefix = ""
            self.move_topic = "/move_action"

        self.action_client = ActionClient(self, MoveGroup, self.move_topic)
        self.action_client.wait_for_server()

        self.latest_eef_pose: np.ndarray = np.eye(4)
        self.base_frame = self.prefix + base_frame
        self.tool_frame = self.prefix + tool_frame

    def get_eef_pose(self):
        """
        Get the latest end effector pose
        The method uses ROS tf client to get the latest transformation data.
        If ROS tf tree doesn't have the required frame, implement the method from keeping arguments and return variables same.

        Arguments: None
        Returns: np.ndarray(4x4)
        """
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, self)

        data = np.eye(4)
        wait_time_nsec = 5.0 * 1e9
        start_time = self.get_clock().now().nanoseconds
        while rclpy.ok() and \
              (self.get_clock().now().nanoseconds - start_time) < wait_time_nsec:
            rclpy.spin_once(self)
            try:
                t = tf_buffer.lookup_transform(
                    self.base_frame, self.tool_frame, Time()
                )
                data[:3,3] = np.array([
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z,
                ])
                data[:3,:3] = R.from_quat([
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w
                ]).as_matrix()
                
                self.latest_eef_pose = data
                break
            except:
                pass
            
        return self.latest_eef_pose

    def move_to_pose(
        self,
        position,
        orientation,
        max_velocity_scaling_factor=0.3,
        max_acceleration_scaling_factor=0.3,
    ):

        position = np.array(position)
        orientation = np.array(orientation)

        # validate if orientation is a quarternion (4x1) or a rotation matrix (3x3)
        if orientation.shape == (3, 3):
            rotation = R.from_matrix(orientation)
            orientation = rotation.as_quat()
        elif orientation.shape == (4,):
            pass
        else:
            return False

        goal = _PoseGoal(
            frame_id=self.base_frame,
            position=position,
            orientation=orientation,
            velocity=max_velocity_scaling_factor,
            acceleration=max_acceleration_scaling_factor,
            method="LIN",
        )

        pose_action = self._create_pose_action(goal)
        self._run_action(pose_action)
        
        return self.get_eef_pose()

    def _create_pose_action(self, goal: _PoseGoal) -> MoveGroup.Goal:
        target_pose = PoseStamped()
        target_pose.header.frame_id = goal.frame_id
        target_pose.pose.position.x = goal.position[0]
        target_pose.pose.position.y = goal.position[1]
        target_pose.pose.position.z = goal.position[2]
        target_pose.pose.orientation.x = goal.orientation[0]
        target_pose.pose.orientation.y = goal.orientation[1]
        target_pose.pose.orientation.z = goal.orientation[2]
        target_pose.pose.orientation.w = goal.orientation[3]
        goal_msg = MoveGroup.Goal()

        request = MotionPlanRequest()
        # request.group_name = self.prefix + "ur20manipulator"
        request.group_name = f"{self.ns}_manipulator"

        request.pipeline_id = "pilz_industrial_motion_planner"
        request.planner_id = goal.method

        request.max_velocity_scaling_factor = goal.velocity
        request.max_acceleration_scaling_factor = goal.acceleration

        pos_constraint = PositionConstraint()
        pos_constraint.header = target_pose.header
        pos_constraint.link_name = self.prefix + "tool0"
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.0001]
        bounding_volume = BoundingVolume()
        bounding_volume.primitives.append(primitive) # type: ignore
        bounding_volume.primitive_poses.append(target_pose.pose) # type: ignore
        pos_constraint.constraint_region = bounding_volume
        pos_constraint.weight = 1.0
        ori_constraint = OrientationConstraint()
        ori_constraint.header = target_pose.header
        ori_constraint.link_name = self.prefix + "tool0"
        ori_constraint.orientation = target_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.01
        ori_constraint.absolute_y_axis_tolerance = 0.01
        ori_constraint.absolute_z_axis_tolerance = 0.01
        ori_constraint.weight = 1.0
        constraint = Constraints()
        constraint.position_constraints.append(pos_constraint) # type: ignore
        constraint.orientation_constraints.append(ori_constraint) # type: ignore
        request.goal_constraints.append(constraint) # type: ignore
        goal_msg.request = request
        return goal_msg

    def _run_action(self, goal_msg):
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        # check if accepted
        if not goal_handle.accepted: # type: ignore
            self.get_logger().error("Goal was rejected by MoveIt! Aborting script.")
            return
        # Wait for the trajectory execution to finish
        self.get_logger().info("Goal accepted. Waiting for completion...")
        result_future = goal_handle.get_result_async() # type: ignore
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        # Verify Success (MoveItErrorCode 1 == SUCCESS)
        if result.error_code.val != 1:
            self.get_logger().error(
                f"Motion failed with error code: {result.error_code.val}. Aborting script."
            )
            return
        self.get_logger().info("Step completed successfully.\n")

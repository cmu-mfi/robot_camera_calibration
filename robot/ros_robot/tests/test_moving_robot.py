from robot.ros_robot.ros_robot import ROSRobot
import numpy as np


robot = ROSRobot(robot_name='yk_builder', rosmaster_ip='172.26.179.142')

position = [0.35, -0.072, 0.605]

orientation = [8.472, 0.707, 0.707, 2.042]

pose = robot.get_eef_pose()

breakpoint()

# robot.move_to_pose(position, orientation)

joint_values = [0.0, 0.0, 0.0, 0.0, -1.57, 0.0]

# robot.move_to_joints(joint_values)
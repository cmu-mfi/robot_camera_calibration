import numpy as np
from scipy.spatial.transform import Rotation as R

from robot.ros2_robot.ros2_robot import Ros2Robot

# TEST1: GET EEF POSE.
def test1(robot):
    pose = robot.get_eef_pose()
    
    rot_m = pose[:3,:3]
    rot = R.from_matrix(rot_m)
    
    print("Current end effector pose: ")
    print(pose)
    
    print(f'rotation as matrix: {rot_m}')
    print(f'rotation as quat: {rot.as_quat()}')
    print(f'rotation as euler xyz: {rot.as_euler('xyz')}')
    print(f'rotation as rot vec: {rot.as_rotvec()}')

# TEST2: MOVE TO POSE.
def test2(robot):
    
    current_pose = robot.get_eef_pose()
    pos = current_pose[:3, 3]
    quart = R.from_matrix(current_pose[:3, :3]).as_quat()
    
    random_delta_pos = np.random.uniform(-0.05, 0.05, size=(3,))
    random_delta_quart = np.random.uniform(-0.1, 0.1, size=(4,))                    
    robot.move_to_pose(position = pos + random_delta_pos,
                       orientation = quart + random_delta_quart)
    
if __name__ == "__main__":

    print("Testing ROS2 Robot class")

    robot_namespace = input("Enter robot namepace: ")
    robot = Ros2Robot(robot_namespace=robot_namespace)

    print("Available tests: ")
    print("1. Get end effector pose")
    print("2. Move to pose")
    test = input("Enter test number (1-2): ")

    match test:
        case "1":
            test1(robot)
        case "2":
            test2(robot)
        case _:
            print("Invalid test number")

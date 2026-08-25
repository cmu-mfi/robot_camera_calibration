import argparse
from calibration.calibrations import *
from calibration.marker.aruco_marker import ArucoMarker
# from camera.orbbec.ob_camera import OBCamera
# from camera.zed_ros.zed_ros import ZedRos
from robot.ros2_robot.ros2_robot import Ros2Robot
from camera.camera_ros2.camera_ros2 import CameraRos2
import numpy as np
import pprint

def robot_camera_calibration():

    camera = CameraRos2(camera_namespace='/bed_side_cam/cam_watch')

    print("=====================================")
    print("CAMERA INITIALIZED")
    print("=====================================")

    marker = ArucoMarker(type='DICT_4X4_100', size=0.05)

    print("=====================================")
    print("MARKER INITIALIZED")
    print("=====================================")

    robot = Ros2Robot(robot_namespace="ur20", base_frame="base")

    print("=====================================")
    print("ROBOT INITIALIZED")
    print("=====================================")
    T_eef2marker = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.050],
            [0.0, 0.0, 1.0, 0.077550],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    '''
    T_eef2marker = np.array(
     [
         [0.0, 1.0, 0.0, -0.050],
         [-1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.077550],
         [0.0, 0.0, 0.0, 1.0],
     ]
    )
    '''
    # camera_matrix = camera.get_rgb_intrinsics()
    # camera_distortion = camera.get_rgb_distortion()
    # image = camera.get_rgb_image()
    # breakpoint()
    # transforms, ids = marker.get_center_poses(input_image = image, 
    #                                         camera_matrix = camera_matrix, 
    #                                         camera_distortion = camera_distortion,
    #                                         depth_image = None)
    # print(np.round(transforms[0],3))
    # breakpoint()
    
    T_robot2camera = get_robot_camera_tf(
        camera, robot, marker, T_eef2marker, 'PLAY', use_depth=False)

    print("=====================================")
    print("RESULTS ARE HERE!!")
    print("=====================================")

    pprint.pprint("T_robot2camera:\n", T_robot2camera)


if __name__ == "__main__":
    robot_camera_calibration()

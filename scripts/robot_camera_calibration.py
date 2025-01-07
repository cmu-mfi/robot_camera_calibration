import argparse
from calibration.calibrations import *
from calibration.marker.aruco_marker import ArucoMarker
from camera.orbbec.ob_camera import OBCamera
from camera.zed_ros.zed_ros import ZedRos
from robot.ros_robot.ros_robot import ROSRobot


def robot_camera_calibration():

    camera = ZedRos(camera_node=f'/cam2/zed_cam2', camera_type='zedxm', rosmaster_ip='localhost')

    print("=====================================")
    print("CAMERA INITIALIZED")
    print("=====================================")

    marker = ArucoMarker(type='DICT_4X4_100', size=0.05)

    print("=====================================")
    print("MARKER INITIALIZED")
    print("=====================================")

    robot = ROSRobot(robot_name='yk_builder', rosmaster_ip='172.26.179.142')

    print("=====================================")
    print("ROBOT INITIALIZED")
    print("=====================================")

    T_eef2marker = np.array(
        [
            [0.0, 0.0, 1.0, 0.041502],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.080824],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    T_robot2camera = get_robot_camera_tf(
        camera, robot, marker, T_eef2marker, 'PLAY', use_depth=False)

    print("=====================================")
    print("RESULTS ARE HERE!!")
    print("=====================================")

    print("T_robot2camera:\n", T_robot2camera)


if __name__ == "__main__":
    robot_camera_calibration()

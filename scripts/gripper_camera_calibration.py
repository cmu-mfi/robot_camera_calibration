import argparse
import os
import cv2

from calibration.calibrations import *
from calibration.marker.aruco_marker import ArucoMarker
from camera.orbbec.ob_camera import OBCamera
from camera.rs_ros.rs_ros import RsRos
from camera.zed_ros.zed_ros import ZedRos
from robot.ros_robot.ros_robot import ROSRobot

def robot_camera_calibration():

    scene_camera = ZedRos(camera_node=f'/cam2/zed_cam2', camera_type='zedxm', rosmaster_ip='localhost')
    scene_calib_file = '/home/mfi/repos/rtc_vision_toolbox/data/calibration_data/zed/T_base2cam2.npy'
    
    inhand_camera = RsRos(camera_node='/camera/realsense2_camera', camera_type='d405', rosmaster_ip='localhost')

    print("=====================================")
    print("CAMERAS INITIALIZED")
    print("=====================================")

    marker = ArucoMarker(type='DICT_4X4_100', size=0.05)

    print("=====================================")
    print("MARKER INITIALIZED")
    print("=====================================")

    robot = ROSRobot(robot_name='yk_builder', rosmaster_ip='172.26.179.142')

    print("=====================================")
    print("ROBOT INITIALIZED")
    print("=====================================")

    scene_image = scene_camera.get_rgb_image()
    scene_camera_matrix = scene_camera.get_rgb_intrinsics()
    scene_camera_distortion = scene_camera.get_rgb_distortion()
    T_scene_camera2marker, ids = marker.get_center_poses(scene_image, scene_camera_matrix, scene_camera_distortion, debug=True)
    T_scene_camera2marker = np.asarray(T_scene_camera2marker[0])

    T_base2scene_camera = np.load(scene_calib_file)
    print(f"T_scene_camera2marker:\n {T_scene_camera2marker}")
    print(f"T_base2scene_camera:\n {T_base2scene_camera}")
    T_base2marker = np.dot(T_base2scene_camera, T_scene_camera2marker)

    print(f"T_base2marker:\n {T_base2marker}")

    print("=====================================")
    print("MARKER IDENTIFIED")
    print("=====================================")

    print("Collecting data for inhand camera")

    T_base2eef_set, T_camera2marker_set = collect_data(
        camera=inhand_camera,
        robot=robot,
        marker=marker,
        method="PLAY",
        num_trials=10,
        verbose=True,
        use_depth=False,
    )
    
    now = datetime.now().strftime("%Y%m%d%H%M")
    np.save("T_camera2marker_set"+ now +".npy", T_camera2marker_set)
    np.save("T_base2eef_set"+ now +".npy", T_base2eef_set)

    print("=====================================")
    print("DATA COLLECTED")
    print("=====================================")
    
    print("Calibrating inhand camera")
    
    methods = ["ONE_SAMPLE_ESTIMATE", "SVD_ALGEBRAIC", "CALIB_HAND_EYE_TSAI", "CALIB_HAND_EYE_ANDREFF"]

    T_eef2marker_set = [np.dot(np.linalg.inv(T_base2eef), T_base2marker) for T_base2eef in T_base2eef_set]
    T_eef2camera_set = []
    avg_error_set = []
    for method in methods:
        print(f"\nMETHOD: {method}")
        T_eef2camera = solve_rigid_transformation(T_eef2marker_set, T_camera2marker_set, method=method)
        avg_error, std_error = calculate_reprojection_error(T_eef2marker_set, T_camera2marker_set, T_eef2camera)
        T_eef2camera_set.append(T_eef2camera)
        avg_error_set.append(avg_error)
        print(f"Transformation matrix T_base2camera:\n{T_eef2camera}")
        print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")

    T_eef2camera = T_eef2camera_set[np.argmin(avg_error_set)]
    np.save("T_eef2camera_"+ now +".npy", T_eef2camera)

    print("=====================================")
    print("RESULTS ARE HERE!!")
    print("=====================================")

    print("T_eef2camera:\n", T_eef2camera)

if __name__ == "__main__":
    robot_camera_calibration()

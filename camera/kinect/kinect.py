import os
import time
from datetime import datetime
from typing import Any, Optional, Union

import cv2
import numpy as np
import open3d as o3d
from pyk4a import PyK4A
from pyk4a.calibration import CalibrationType

class KinectCamera:
    """
    This class represents a camera object for capturing RGB, depth and pointcloud data.
    """
    def __init__(self, device_id: int=0, debug: bool=False):
        """
        Initializes the Camera object.
        """
        self.debug = debug
        self.k4a = PyK4A(device_id=device_id)
        self.k4a.start()
        if self.debug:
            print("Started Azure Kinect camera with device ID: ", device_id)

    def __del__(self):
        """
        Deinitializes the Camera object.
        """
        self.k4a.stop()
        if self.debug:
            print("Stopped Azure Kinect camera")

    def get_rgb_image(self, use_new_frame: bool=True) -> Union[Optional[np.array], Any]:
        """
        Captures and returns an RGB image from the camera.
        """
        capture = self.k4a.get_capture()
        if capture.color is not None:
            color_image = capture.color[:, :, :3]
            return color_image
        return None

    def get_raw_depth_data(self, max_depth=None, use_new_frame: bool=True, method=None) -> Optional[np.ndarray]:
        """
        Captures and returns raw depth data from the camera.

        Args:
            max_depth (int): Maximum depth value to include in the colormap in mm
        
        Returns:
            depth_data (numpy.ndarray): Raw depth data (in mm) captured from the camera.
        """
        capture = self.k4a.get_capture()
        if capture.depth is not None:
            depth_data = capture.depth.astype(np.float32)  # Keep depth in mm
            depth_data[depth_data == float('inf')] = np.nan
            depth_data[depth_data == float('-inf')] = np.nan

            if max_depth is not None:
                depth_data = np.where(depth_data < max_depth, depth_data, np.nan)

            return depth_data
        return None

    def get_depth_image(self, max_depth=None, use_new_frame: bool=True, method=None) -> Optional[np.ndarray]:
        """
        Captures and returns a raw depth image from the camera.

        Args:
            max_depth (int): Maximum depth value to include in the colormap in mm
        
        Returns:
            depth_image (numpy.ndarray): Raw depth image captured from the camera.
        """
        depth_data = self.get_raw_depth_data(max_depth=max_depth)
        if depth_data is not None:
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return depth_image
        return None

    def get_colormap_depth_image(self, min_depth: int=20, max_depth: int=10000,
                                 colormap: int=cv2.COLORMAP_JET, use_new_frame: bool=True, method=None) -> Optional[np.ndarray]:
        """
        Captures and returns a depth image with a colormap applied.

        Args:
            min_depth (int): Minimum depth value to include in the colormap.
            max_depth (int): Maximum depth value to include in the colormap.
            colormap (int): OpenCV colormap to apply to the depth image.

        Returns:
            depth_image (numpy.ndarray): Depth image with colormap applied.
        """
        depth_data = self.get_raw_depth_data()
        if depth_data is not None:
            depth_data = np.where((depth_data > min_depth) & (depth_data < max_depth), depth_data, 0)
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, colormap)
            return depth_image
        return None

    def get_point_cloud(self, min_mm: int = 40, max_mm: int = 1000,
                        save_points: bool = False, use_new_frame: bool = True, method=None) -> Optional[o3d.geometry.PointCloud]:
        """
        Captures and returns a point cloud from the camera.

        Returns:
            point_cloud (o3d.geometry.PointCloud): Point cloud captured from the camera.
        """
        capture = self.k4a.get_capture()
        if capture.depth is not None and capture.color is not None:
            points = capture.depth_point_cloud
            points = points.reshape(-1, 3)

            colors = capture.transformed_color 
            colors = colors.reshape(-1, 4)[:, :3] / 255.0
            colors = colors[:, [2, 1, 0]]
            colors = colors.astype(np.float32)

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            if save_points:
                now = datetime.now().strftime("%m%d_%H%M")
                points_filename = f"points_{now}.ply"
                o3d.io.write_point_cloud(points_filename, point_cloud)
            return point_cloud
        return None

    def get_rgb_intrinsics(self) -> Optional[np.ndarray]:
        """
        Returns the RGB camera matrix.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]

        Returns:
            camera_matrix: Intrinsics of the camera as camera matrix.
        """
        return self.k4a.calibration.get_camera_matrix(CalibrationType.COLOR)

    def get_depth_intrinsics(self) -> Optional[np.ndarray]:
        """
        Returns the depth camera matrix.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]

        Returns:
            camera_matrix: Intrinsics of the camera as camera matrix.
        """
        return self.k4a.calibration.get_camera_matrix(CalibrationType.DEPTH)

    def get_rgb_distortion(self) -> Optional[np.ndarray]:
        """
        Returns the RGB camera distortion coefficients.

        Returns:
            distortion: Distortion coefficients of the camera.
        """
        return self.k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    def get_depth_distortion(self) -> Optional[np.ndarray]:
        """
        Returns the depth camera distortion coefficients.

        Returns:
            distortion: Distortion coefficients of the camera.
        """
        return self.k4a.calibration.get_distortion_coefficients(CalibrationType.DEPTH)
    
    def visualize_point_cloud(self):
        """
        Visualizes the point cloud and the origin reference.
        """
        pointcloud = self.get_point_cloud()
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])

        # Add reference frame to the visualization
        o3d.visualization.draw_geometries(
            [camera_frame, pointcloud],
            lookat=pointcloud.get_center(),
            up=np.array([0.0, -1.0, 0.0]),
            front=-pointcloud.get_center(),
            zoom=1
        )

    def close(self):
        """
        Stops the camera.
        """
        self.k4a.stop()
        
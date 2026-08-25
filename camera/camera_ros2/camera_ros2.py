import base64
import math
import struct
import sys
import threading
import time
from ctypes import POINTER, c_float, c_uint32, cast, pointer
from datetime import datetime
from typing import Any, Optional, Union

import cv2
import numpy as np
import open3d as o3d
import rclpy
import torch
from igev_stereo import IGEVStereo
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from utils.utils import InputPadder

# Topic Constants
# RGB_IMAGE = 'color/image_raw'
# RGB_INFO_TOPIC = 'color/camera_info'
RGB_IMAGE = 'infra1/image_rect_raw'
RGB_INFO_TOPIC = 'infra1/camera_info'
LEFT_IMAGE = 'infra1/image_rect_raw'
LEFT_IMAGE_INFO_TOPIC = 'infra1/camera_info'
RIGHT_IMAGE = 'infra2/image_rect_raw'
RIGHT_IMAGE_INFO_TOPIC = 'infra2/camera_info'

RGB_IMAGE = LEFT_IMAGE
RGB_INFO_TOPIC = LEFT_IMAGE_INFO_TOPIC

RGB_COMPRESSED_TOPIC = 'rgb/image_rect_color/compressed'
DEPTH_TOPIC = 'depth/depth_registered'
POINTCLOUD_TOPIC = 'point_cloud/cloud_registered'
DEPTH_CAMERA_INFO_TOPIC = 'depth/camera_info'


class CameraRos2(Node):
    def __init__(self, camera_namespace: str = '', depth_mode='igev', debug=False, camera_type='d555'):
        # Passing namespace here handles the "namespace/topic" logic automatically
        if not rclpy.ok():
            rclpy.init()        
        
        super().__init__('camera_client_node', namespace=camera_namespace)
        
        self.debug = debug
        self.__depth_mode = depth_mode
        self.__camera_type = camera_type
        
        # Internal storage for the "latched" frames
        self._current_msg = {
            'rgb': None,
            'rgb_info': None,
            'left_image': None,
            'right_image': None,
            'left_info': None,
            'depth': None,
            'pcd': None,
            'info': None
        }

        # Setup Subscriptions (using relative names)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        
        self.create_subscription(Image, RGB_IMAGE, lambda msg: self._cb('rgb', msg), qos)
        self.create_subscription(Image, LEFT_IMAGE, lambda msg: self._cb('left_image', msg), qos)
        self.create_subscription(Image, RIGHT_IMAGE, lambda msg: self._cb('right_image', msg), qos)
        self.create_subscription(CameraInfo, RGB_INFO_TOPIC, lambda msg: self._cb('rgb_info', msg), qos)
        self.create_subscription(CameraInfo, LEFT_IMAGE_INFO_TOPIC, lambda msg: self._cb('left_info', msg), qos)
        self.create_subscription(Image, DEPTH_TOPIC, lambda msg: self._cb('depth', msg), qos)
        self.create_subscription(PointCloud2, POINTCLOUD_TOPIC, lambda msg: self._cb('pcd', msg), qos)
        
        # self.__start_executor()

    def _cb(self, key, msg):
        self._current_msg[key] = msg

    def _wait_for_msg(self, key, timeout=5.0):
        start = self.get_clock().now()
        while self._current_msg[key] is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout:
                raise TimeoutError(f"Timed out waiting for {key} topic.")
        return self._current_msg[key]

    def get_rgb_image(self):
        self._current_msg['rgb'] = None  # Clear the current message to ensure we get a new one
        msg = self._wait_for_msg('rgb')
        return self.__get_image_from_rostopic(msg)
    
    def get_rgb_intrinsics(self):
        self._current_msg['rgb_info'] = None  # Clear the current message to ensure we get a new one
        msg: CameraInfo = self._wait_for_msg('rgb_info') # type: ignore
        return msg.k.reshape(3, 3)

    def get_rgb_distortion(self):
        self._current_msg['rgb_info'] = None  # Clear the current message to ensure we get a new one
        msg: CameraInfo = self._wait_for_msg('rgb_info') # type: ignore
        return np.asarray(msg.d)

    def get_infra_intrinsics(self):
        self._current_msg['left_info'] = None  # Clear the current message to ensure we get a new one
        msg: CameraInfo = self._wait_for_msg('left_info') # type: ignore
        return msg.k.reshape(3, 3)

    def get_infra_distortion(self):
        self._current_msg['left_info'] = None  # Clear the current message to ensure we get a new one
        msg: CameraInfo = self._wait_for_msg('left_info') # type: ignore
        return np.array(msg.d)    

    def __get_image_from_rostopic(self, img_msg):
        """Manual conversion from sensor_msgs/Image to numpy without cv_bridge."""
        
        def encoding_to_dtype_with_channels(encoding):
            mapping = {
                'mono8': (np.uint8, 1), 'bgr8': (np.uint8, 3), 'rgb8': (np.uint8, 3),
                'mono16': (np.uint16, 1), '32FC1': (np.float32, 1)
            }
            if encoding not in mapping:
                raise TypeError(f'Unsupported encoding: {encoding}')
            return mapping[encoding]

        dtype, n_channels = encoding_to_dtype_with_channels(img_msg.encoding)
        
        # ROS2 data is already a bytes-like object, no base64 needed
        im = np.frombuffer(img_msg.data, dtype=dtype).reshape(
            img_msg.height, img_msg.width, n_channels if n_channels > 1 else 1
        )

        if n_channels == 1:
            im = im.reshape(img_msg.height, img_msg.width)

        # Handle Byte Order
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            im = im.byteswap().newbyteorder() # type: ignore

        # Color Conversion
        if img_msg.encoding == 'bgr8':
            # Convert BGR (OpenCV/ROS default) to RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif img_msg.encoding == 'rgb8':
            # Already RGB, but we for some reason it is BGR
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
            pass
        elif img_msg.encoding == 'mono8' or img_msg.encoding == 'mono16':
            # Optionally convert grayscale to RGB if your pipeline expects 3 channels
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        
        return im

# above this all good
# below to be implemented

    def get_raw_depth_data(self, max_depth=None, use_new_frame: bool=True, method=None):
        """
        Captures and returns a raw depth image from the camera.

        Args:
            max_depth (int): Maximum depth value to include in the colormap in mm
            TODO: use_new_frame (bool): If True, captures a new frame. If False, uses the latest frame.
        
        Returns:
            depth_data (numpy.ndarray): Raw depth data (in mm) captured from the camera.
        """
        depth_data:np.ndarray = None
        
        if method is None:
            method = self.__depth_mode

        if method == 'default':
            depth_data = self.__get_image_from_rostopic('/depth/image_rect_raw')
            # convert to mm
            depth_data = depth_data.astype(np.float32) * 1000

        elif method == 'igev':
            depth_data = self.__estimate_depth_IGEV()

        # replace +-inf values with nan
        depth_data = depth_data.copy()
        depth_data[depth_data == float('inf')] = np.nan
        depth_data[depth_data == float('-inf')] = np.nan

        # convert meters to mm
        if method == 'igev':
            depth_data = depth_data * 1000
            
        if max_depth is not None:
            depth_data = np.where(depth_data < max_depth, depth_data, np.nan)

        return depth_data

    def get_depth_image(self, max_depth = None, use_new_frame: bool=True, method=None):
        """
        Captures and returns a raw depth image from the camera.

        Args:
            max_depth (int): Maximum depth value to include in the colormap in mm
            TODO: use_new_frame (bool): If True, captures a new frame. If False, uses the latest frame.
        
        Returns:
            depth_image (numpy.ndarray): Raw depth image captured from the camera.
        """

        if method is None:
            method = self.__depth_mode

        depth_data = self.get_raw_depth_data(max_depth=max_depth, method=method, use_new_frame=use_new_frame)

        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return depth_image

    def get_colormap_depth_image(self, min_depth: int=20, max_depth: int=10000,
                                 colormap: int=cv2.COLORMAP_JET, use_new_frame: bool=True, method=None):
        """
        Captures and returns a depth image with a colormap applied.

        Args:
            min_depth (int): Minimum depth value to include in the colormap.
            max_depth (int): Maximum depth value to include in the colormap.
            colormap (int): OpenCV colormap to apply to the depth image.
            TODO: use_new_frame (bool): If True, captures a new frame. If False, uses the latest frame.

        Returns:
            depth_image (numpy.ndarray): Depth image with colormap applied.
        """

        if method is None:
            method = self.__depth_mode

        depth_image = self.get_raw_depth_data(method=method, use_new_frame=use_new_frame)        
        depth_image = np.where((depth_image > min_depth) & (depth_image < max_depth),
                            depth_image, 0)
        depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_image = cv2.applyColorMap(depth_image, colormap)

        return depth_image

    def get_point_cloud(self, min_mm: int = 40, max_mm: int = 1000,
                        save_points: bool = False, method=None) -> o3d.geometry.PointCloud:
        """
        Captures and returns a point cloud from the camera.

        Returns:
            point_cloud (o3d.geometry.PointCloud): Point cloud captured from the camera.
        """
        if method is None:
            method = self.__depth_mode

        point_cloud = None
        if method == 'default':
            point_cloud = self.__get_pointcloud_from_rostopic('/point_cloud/cloud_registered')

        elif method == 'igev':
            point_cloud = self.__estimate_pcd_IGEV(min_mm, max_mm)

        if (save_points):
            now = datetime.now().strftime("%m%d_%H%M")
            points_filename = f"points_{self.__camera_type}_{method}_{now}.ply"
            o3d.io.write_point_cloud(points_filename, point_cloud)

        return point_cloud


    def __get_pointcloud_from_rostopic(self, pcd_msg):
        """Manual conversion from PointCloud2 to Open3D."""
        # Field unpacking logic
        field_names = [f.name for f in pcd_msg.fields]
        
        # Using a generator for memory efficiency (same as your __read_points)
        cloud_data = list(self.__read_points(pcd_msg, field_names=field_names, skip_nans=True))
        
        if not cloud_data:
            return None

        open3d_cloud = o3d.geometry.PointCloud()
        xyz = np.array([(p[0], p[1], p[2]) for p in cloud_data])
        
        # Scaling to mm if that is your project standard
        open3d_cloud.points = o3d.utility.Vector3dVector(xyz * 1000.0)

        if 'rgb' in field_names:
            # Re-implementing your bit-shift logic for packed RGB
            rgb = []
            for p in cloud_data:
                # RGB is usually at index 3 if present
                packed = p[3]
                if isinstance(packed, float):
                    packed = struct.unpack('I', struct.pack('f', packed))[0]
                r = (packed & 0x00FF0000) >> 16
                g = (packed & 0x0000FF00) >> 8
                b = (packed & 0x000000FF)
                rgb.append((r/255.0, g/255.0, b/255.0))
            open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb))

        return open3d_cloud

    def __read_points(self, cloud, field_names=None, skip_nans=False):
        """ROS2 version of the point reader."""
        # Datatype mapping from sensor_msgs/PointField
        _DATATYPES = {1:('b',1), 2:('B',1), 3:('h',2), 4:('H',2), 5:('i',4), 6:('I',4), 7:('f',4), 8:('d',8)}

        fmt = ">" if cloud.is_bigendian else "<"
        offset = 0
        for field in sorted(cloud.fields, key=lambda f: f.offset):
            if offset < field.offset:
                fmt += "x" * (field.offset - offset)
                offset = field.offset
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

        unpack_from = struct.Struct(fmt).unpack_from
        
        # ROS2 data is directly accessible as bytes
        data = cloud.data 
        
        for v in range(cloud.height):
            row_offset = cloud.row_step * v
            for u in range(cloud.width):
                p = unpack_from(data, row_offset + (u * cloud.point_step))
                if skip_nans and any(np.isnan(val) for val in p if isinstance(val, float)):
                    continue
                yield p
    
    def __estimate_depth_IGEV(self):
        '''
        Estimate depth using IGEV generated disparity map.
        '''
        disparity = self.__estimate_disp_IGEV()
        unified_matrix = self.get_infra_intrinsics()
        args = self.__get_default_IGEV_args()
        camera_seperation = args.camera_seperation

        # camera intrinsics are for 480p, so we need to scale them to the current image size
        focal_length = unified_matrix[0][0] * disparity.shape[0] / 480

        depth = (camera_seperation * focal_length) / disparity

        return depth

    def __estimate_disp_IGEV(self):
        """
        Get disparity map using IGEV method.
        """
        left_image, right_image = self.__get_stereo_images()
        
        if self.debug:
            print("::::::starting IGEV estimation::::::")

        args = self.__get_default_IGEV_args()        
        model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
        model.load_state_dict(torch.load(args.restore_ckpt))
        model = model.module
        model.to('cuda')
        model.eval()

        with torch.no_grad():
            image1 = left_image
            image1_np = np.array(image1[:,:,0:3])
            image2 = right_image
            image2_np = np.array(image2[:,:,0:3])

            # # downsampel to 720p to speed up
            # image1_np = cv2.resize(image1_np,(1280,720))
            # image2_np = cv2.resize(image2_np,(1280,720))

            # preprocess FOR igev
            image1= image1_np.astype(np.uint8)
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
            image1 = image1[None].to('cuda')
            image2= image2_np.astype(np.uint8)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
            image2 = image2[None].to('cuda')

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            start = time.time()

            igev_disp, disp_prob = model(image1, image2, iters=32, test_mode=True)
            end = time.time()
            if self.debug:
                print("torch inference time: ", end - start)
            igev_disp = igev_disp.cpu().numpy()
            igev_disp = padder.unpad(igev_disp)
            igev_disp = igev_disp.squeeze()
            disp_prob = disp_prob.cpu().numpy()
            disp_prob = padder.unpad(disp_prob)
            disp_prob = disp_prob.squeeze()
            disp_prob = np.float32( disp_prob / 4 )
            
            igev_disp = np.float32( igev_disp )    
            
            # remove disparity values whose probability is less than 0.5
            # igev_disp[disp_prob < 0.5] = 0
            igev_disp[np.where(igev_disp == 0)] = None      
            igev_disp[np.where(igev_disp == float('inf'))] = None
            
            np.save("disp_prob.npy", disp_prob)
            
            if self.debug:
                print("::::::IGEV estimation finished::::::")
                print("disp_prob: ", disp_prob.shape)
                print("IGEV disp shape: ", igev_disp.shape)

            return igev_disp

    def __estimate_pcd_IGEV(self, min_mm: int = 40, max_mm: int = 10000, 
                            use_new_frame: bool=True) -> o3d.geometry.PointCloud:
        """
        Estimate point cloud using IGEV generated disparity map.
        """
        igev_disp = self.__estimate_disp_IGEV()
        args = self.__get_default_IGEV_args()

        # set Q matrix
        imageSize = (720, 1280)
        R = np.eye(3)
        T = np.array([-args.camera_seperation, 0, 0])
        
        # camera intrinsics are for 480p, so we need to scale them to the current image size        
        cameraMatrix = self.get_infra_intrinsics() * igev_disp.shape[0] / 480
        distCoeffs = self.get_infra_distortion()
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix, distCoeffs, 
                                                                          cameraMatrix, distCoeffs, 
                                                                          imageSize, R, T)

        with torch.no_grad():
            # rgb point cloud, reference : https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
            disp = igev_disp
            image_3d = cv2.reprojectImageTo3D(disp, Q)

            points = []
            for i in range( image_3d.shape[0] ):
                for j in range(image_3d.shape[1]):
                    x = image_3d[i][j][0]*1000
                    y = image_3d[i][j][1]*1000
                    z = image_3d[i][j][2]*1000
                    pt = [x, y, z]
                    points.append(pt)
            if self.debug:
                print("finished")   

            points = np.array(points)
            new_points = []
            for point in points:
                if point[-1] < max_mm and point[-1] > min_mm:
                    new_points.append(point)
            
            if len(new_points) == 0:
                new_points = points        
            
            points = np.array(new_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            return pcd

    def __get_stereo_images(self):
        """
        Get stereo image from ROS topic.
        """
        # GET RIGHT AND LEFT IMAGES
        self._current_msg['left_image'] = None  # Clear the current message to ensure we get new ones
        self._current_msg['right_image'] = None
        left_msg = self._wait_for_msg('left_image')
        right_msg = self._wait_for_msg('right_image')
        left_image = self.__get_image_from_rostopic(left_msg)
        right_image = self.__get_image_from_rostopic(right_msg)
        
        if left_image is None or right_image is None:
            raise Exception('Failed to get stereo images from ROS topics.')

        # stereo_image = self.__get_image_from_rostopic('/stereo/image_rect_color')

        # # GET RIGHT AND LEFT IMAGES
        # left_image = stereo_image[:, :int(stereo_image.shape[1]/2)]
        # right_image = stereo_image[:, int(stereo_image.shape[1]/2):]

        return left_image, right_image

    def __get_default_IGEV_args(self):
       
        class args:
            restore_ckpt = '/home/shobhit/repos/robot_camera_calibration/camera/camera_ros2/middlebury.pth'
            save_numpy = False
            left_imgs = "./demo-imgs/*/im0.png"
            right_imgs = "./demo-imgs/*/im1.png"
            output_directory = "./demo-output/"
            mixed_precision = False
            valid_iters = 32
            hidden_dims = [128]*3
            corr_implementation = "reg"
            shared_backbone = False
            corr_levels = 2
            corr_radius = 4
            n_downsample = 2
            slow_fast_gru = False
            n_gru_layers = 3
            max_disp = 192
            downsampling = False
            camera_seperation = 0.12
            
        if self.__camera_type == 'd405':
            args.camera_seperation = 0.018
        elif self.__camera_type == 'd555':
            args.camera_seperation = 0.09503433853387833

        return args

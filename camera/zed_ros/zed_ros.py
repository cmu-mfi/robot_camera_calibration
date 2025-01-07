import os
import argparse
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
import roslibpy
import torch
from igev_stereo import IGEVStereo
from utils.utils import InputPadder


class ZedRos:
    """
    This class represents a camera object for capturing RGB, depth and pointcloud data.
    """
    """
    Methods:
        __init__(serial_no=None): Initializes the Camera object.
        get_rgb_image(): Captures and returns an RGB image from the camera.
        get_raw_depth_image(): Captures and returns a raw depth image from the camera.
        get_colormap_depth_image(min_depth=20, max_depth=10000, colormap=cv2.COLORMAP_JET): Captures and returns a depth image with a colormap applied.
        get_point_cloud(min_mm: int = 400, max_mm: int = 1000, save_points=False): Captures and returns a point cloud from the camera.
        get_rgb_intrinsics(): Returns the camera matrix.
        get_depth_intrinsics(): Returns the depth camera matrix.
        get_rgb_distortion(): Returns the rgb camera distortion coefficients.
        get_depth_distortion(): Returns the depth camera distortion coefficients.
    """

    # data members
    __ros_client    :roslibpy.Ros = None
    __camera_node   :str = None
    __depth_mode    :str = None
    __camera_type   :str = None # 'zedx' or 'zedxm'

    def __init__(self, camera_node: str, 
                 camera_type: str, 
                 rosmaster_ip='localhost', 
                 rosmaster_port=9090, 
                 depth_mode='igev',
                 debug = False):
        """
        Initializes the Camera object.
        """
        
        self.debug = debug
        
        # initialize ros client
        self.__ros_client = roslibpy.Ros(host=rosmaster_ip, port=rosmaster_port)
        self.__ros_client.run()
        print('Is ROS connected?', self.__ros_client.is_connected)      

        if not self.__ros_client.is_connected:
            raise Exception('ROS client is not connected. Please check the ROS master IP and port.')

        # check if camera node exists
        print(f'Getting details of camera node {camera_node}')
        output = self.__ros_client.get_node_details(node = camera_node)
        if len(output['publishing'])==0:
            raise Exception(f'Camera node {camera_node} does not exist. Please check the camera node name.')

        self.__camera_type = camera_type
        self.__camera_node = camera_node
        self.__depth_mode = depth_mode
        # self.__stereo_ros = None
        
        # self.__stereo_thread = threading.Thread(target=self.__ros_sub_stereo)
        # self.__stereo_thread.start()

    def __del__(self):
        # self.__ros_client.terminate()
        pass
    
    # def __ros_sub_stereo(self):
    #     image_subscriber = roslibpy.Topic(self.__ros_client, '/'+self.__camera_node + "/stereo/image_rect_color", 'sensor_msgs/Image')        
    #     image_subscriber.subscribe(self.__stereo_callback)           
    
    # def __stereo_callback(self, message):
    #     self.__stereo_ros = message
    
    def get_rgb_image(self, use_new_frame: bool=True) -> Union[Optional[np.array], Any]:
        """
        Captures and returns an RGB image from the camera.

        Args:
            TODO: use_new_frame (bool): If True, captures a new frame. If False, uses the last saved image.
        
        Returns:
            color_image (numpy.ndarray): RGB image captured from the camera.
        """
        # color_image = self.__get_image_from_rostopic('/rgb/image_rect_color')
        color_image = self.__get_compressedimage_from_rostopic('/rgb/image_rect_color/compressed')
        color_image = np.array(color_image[:,:,0:3])

        return color_image

    def get_raw_depth_data(self, max_depth=None, use_new_frame: bool=True, method=None):
        """
        Captures and returns a raw depth image from the camera.

        Args:
            max_depth (int): Maximum depth value to include in the colormap in mm
            TODO: use_new_frame (bool): If True, captures a new frame. If False, uses the latest frame.
        
        Returns:
            depth_data (numpy.ndarray): Raw depth data (in mm) captured from the camera.
        """
        depth_data :np.array = None

        if method is None:
            method = self.__depth_mode

        if method == 'default':
            depth_data = self.__get_image_from_rostopic('/depth/depth_registered')
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
            depth_data = np.where(depth_data < max_depth, depth_data, None)

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
                        save_points: bool = False, use_new_frame: bool = True, method=None) -> o3d.geometry.PointCloud:
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
            point_cloud = self.__estimate_pcd_IGEV(min_mm, max_mm, use_new_frame)

        if (save_points):
            now = datetime.now().strftime("%m%d_%H%M")
            points_filename = f"points_{self.__camera_node[1:5]}_{method}_{now}.ply"
            o3d.io.write_point_cloud(points_filename, point_cloud)

        return point_cloud

    def get_rgb_intrinsics(self):
        """
        Returns the camera matrix.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]

        Returns:
            camera_matrix: Intrinsics of the camera as camera matrix.
        """

        camera_info_sub = roslibpy.Topic(self.__ros_client, '/'+self.__camera_node + '/rgb/camera_info', 'sensor_msgs/CameraInfo')        

        def callback(message):
            nonlocal camera_matrix
            camera_matrix = np.array(message['K']).reshape(3, 3)

        camera_matrix :np.array = None
        while camera_matrix is None:
            camera_info_sub.subscribe(lambda message: callback(message))
        camera_info_sub.unsubscribe()

        return camera_matrix

    def get_depth_intrinsics(self):
        """
        Returns the camera matrix.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]

        Returns:
            camera_matrix: Intrinsics of the camera as camera matrix.
        """
        camera_info_sub = roslibpy.Topic(self.__ros_client, '/'+self.__camera_node + '/depth/camera_info', 'sensor_msgs/CameraInfo')        

        def callback(message):
            nonlocal camera_matrix
            camera_matrix = np.array(message['K']).reshape(3, 3)

        camera_matrix :np.array = None
        while camera_matrix is None:
            camera_info_sub.subscribe(lambda message: callback(message))
        camera_info_sub.unsubscribe()

        return camera_matrix

    def get_rgb_distortion(self):
        """
        Returns the rgb camera distortion coefficients.

        Returns:
            distortion: Distortion coefficients of the camera.
        """
        camera_info_sub = roslibpy.Topic(self.__ros_client, '/'+self.__camera_node + '/rgb/camera_info', 'sensor_msgs/CameraInfo')        

        def callback(message):
            nonlocal distortion
            distortion = np.array(message['D'])

        distortion :np.array = None
        while distortion is None:
            camera_info_sub.subscribe(lambda message: callback(message))        

        camera_info_sub.unsubscribe()

        return distortion

    def get_depth_distortion(self):
        """
        Returns the depth camera distortion coefficients.

        Returns:
            distortion: Distortion coefficients of the camera.
        """
        camera_info_sub = roslibpy.Topic(self.__ros_client, '/'+self.__camera_node + '/depth/camera_info', 'sensor_msgs/CameraInfo')        

        def callback(message):
            nonlocal distortion
            distortion = np.array(message['D'])

        distortion :np.array = None
        while distortion is None:
            camera_info_sub.subscribe(lambda message: callback(message))

        camera_info_sub.unsubscribe()

        return distortion

    def __get_image_from_rostopic(self, topic_name: str):
        """
        Get image from ROS topic.
        Based on cv_bridge implementation
        https://github.com/ros-perception/vision_opencv/blob/a34a0c261984e4ab71f267d093f6a48820801d80/cv_bridge/python/cv_bridge/core.py#L147
        """

        def encoding_to_dtype_with_channels(encoding):
            if encoding == 'mono8':
                return np.uint8, 1
            elif encoding == 'bgr8':
                return np.uint8, 3
            elif encoding == 'rgb8':
                return np.uint8, 3
            elif encoding == 'mono16':
                return np.uint16, 1
            elif encoding == 'rgba8':
                return np.uint8, 4
            elif encoding == 'bgra8':
                return np.uint8, 4
            elif encoding == '32FC1':
                return np.float32, 1
            elif encoding == '32FC2':
                return np.float32, 2
            elif encoding == '32FC3':
                return np.float32, 3
            elif encoding == '32FC4':
                return np.float32, 4
            else:
                raise TypeError(f'Unsupported encoding: {encoding}')
        
        def callback(img_msg):
            print("Got image!")
            nonlocal image, image_subscriber
            img_msg['data'] = base64.b64decode(img_msg['data'])
            img_msg['data'] = np.frombuffer(img_msg['data'], dtype=np.uint8)
            dtype, n_channels = encoding_to_dtype_with_channels(img_msg['encoding'])
            dtype = np.dtype(dtype)
            dtype = dtype.newbyteorder('>' if img_msg['is_bigendian'] else '<')

            img_buf = np.asarray(img_msg['data'], dtype=dtype) if isinstance(img_msg['data'], list) else img_msg['data']

            if n_channels == 1:
                im = np.ndarray(shape=(img_msg['height'], int(img_msg['step']/dtype.itemsize)),
                                dtype=dtype, buffer=img_buf)
                im = np.ascontiguousarray(im[:img_msg['height'], :img_msg['width']])
            else:
                im = np.ndarray(shape=(img_msg['height'], int(img_msg['step']/dtype.itemsize/n_channels), n_channels),
                                dtype=dtype, buffer=img_buf)
                im = np.ascontiguousarray(im[:img_msg['height'], :img_msg['width'], :])

            # If the byte order is different between the message and the system.
            if img_msg['is_bigendian'] == (sys.byteorder == 'little'):
                im = im.byteswap().newbyteorder()

            image = im
            image_subscriber.unsubscribe()        

        # if topic_name != '/stereo/image_rect_color':
        image_subscriber = roslibpy.Topic(self.__ros_client, self.__camera_node + topic_name, 'sensor_msgs/Image')        

        image :np.ndarray = None
        image_subscriber.subscribe(lambda message: callback(message))
        time_stamp_2 = time.time()
        while image is None:
            continue
            # image_subscriber.subscribe(lambda message: callback(message))

        time_stamp_3 = time.time()
        
        if self.debug:
            print(f"Time taken to get image: {time_stamp_3 - time_stamp_2}s")

        # image_subscriber.unsubscribe()        
    
        # else:
        #     self.__stereo_ros = None
        #     time_stamp_1 = time.time()        
        #     while self.__stereo_ros is None:
        #         pass
        #     time_stamp_2 = time.time()
        #     print(f"Time taken to get image: {time_stamp_2 - time_stamp_1}s")

        #     image :np.ndarray = None
        #     callback(self.__stereo_ros)
            
        return image

    def __get_compressedimage_from_rostopic(self, topic_name: str):
        """
        Get image from ROS topic.
        Based on cv_bridge implementation
        https://github.com/ros-perception/vision_opencv/blob/a34a0c261984e4ab71f267d093f6a48820801d80/cv_bridge/python/cv_bridge/core.py#L147
        """
        
        def callback(img_msg):
            print("Got image!")
            nonlocal image, image_subscriber
            img_msg['data'] = base64.b64decode(img_msg['data'])
            img_msg['data'] = np.frombuffer(img_msg['data'], dtype=np.uint8)
            
            img_buf = np.ndarray(shape=(1, len(img_msg['data'])),
                                 dtype=np.uint8, buffer=img_msg['data'])
            
            image = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
            image_subscriber.unsubscribe()

        image_subscriber = roslibpy.Topic(self.__ros_client, self.__camera_node + topic_name, 'sensor_msgs/CompressedImage')        

        image :np.ndarray = None
        image_subscriber.subscribe(lambda message: callback(message))
        time_stamp_2 = time.time()
        while image is None:
            continue

        time_stamp_3 = time.time()
        
        if self.debug:
            print(f"Time taken to get image: {time_stamp_3 - time_stamp_2}s")
            
        return image


    def __get_pointcloud_from_rostopic(self, topic_name: str):
        """
        Get pointcloud from ROS topic.
        https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/src/sensor_msgs/point_cloud2.py
        https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
        """
        convert_rgbUint32_to_tuple = lambda rgb_uint32: (
            (rgb_uint32 & 0x00FF0000) >> 16,
            (rgb_uint32 & 0x0000FF00) >> 8,
            (rgb_uint32 & 0x000000FF),
        )
        convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
            int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
        )

        print(f"Getting point cloud from ROS topic {self.__camera_node + topic_name}")

        pcd_subscriber = roslibpy.Topic(
            self.__ros_client,
            "/" + self.__camera_node + topic_name,
            "sensor_msgs/PointCloud2",
        )

        def callback(pcd_msg):
            nonlocal pcd
            pcd = pcd_msg
            print("Got pcd!")

        pcd: np.ndarray = None
        while pcd is None:
            pcd_subscriber.subscribe(lambda message: callback(message))

        pcd_subscriber.unsubscribe()

        # Get cloud data from ros_cloud
        field_names = [field["name"] for field in pcd["fields"]]
        cloud_data = list(
            self.__read_points(pcd, skip_nans=True, field_names=field_names)
        )

        # Check empty
        open3d_cloud = o3d.geometry.PointCloud()

        if len(cloud_data) == 0:
            print("Converting an empty cloud")
            return None

        # Set open3d_cloud
        if "rgb" in field_names:
            IDX_RGB_IN_FIELD = 3  # x, y, z, rgb

            # Get xyz
            xyz = [
                (x, y, z) for x, y, z, rgb in cloud_data
            ]  # (why cannot put this line below rgb?)

            # Get rgb
            # Check whether int or float
            if (
                type(cloud_data[0][IDX_RGB_IN_FIELD]) == float
            ):  # if float (from pcl::toROSMsg)
                rgb = [convert_rgbFloat_to_tuple(rgb) for x, y, z, rgb in cloud_data]
            else:
                rgb = [convert_rgbUint32_to_tuple(rgb) for x, y, z, rgb in cloud_data]

            # combine
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz) * 1000)
            open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb) / 255.0)
        else:
            xyz = [(x, y, z) for x, y, z in cloud_data]  # get xyz
            open3d_cloud.points = o3d.Vector3dVector(np.array(xyz))

        # return
        return open3d_cloud

    def __read_points(self, cloud, field_names=None, skip_nans=False, uvs=[]):
        """
        Read points from a L{sensor_msgs.PointCloud2} message.
        Implementation based on code from:
        https://github.com/ros/common_msgs/blob/20a833b56f9d7fd39655b8491a2ec1226d2639b3/sensor_msgs/src/sensor_msgs/point_cloud2.py#L61

        @param cloud: The point cloud to read from.
        @param field_names: The names of fields to read. If None, read all fields. [default: None]
        @type  field_names: iterable
        @param skip_nans: If True, then don't return any point with a NaN value.
        @type  skip_nans: bool [default: False]
        @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
        @type  uvs: iterable
        @return: Generator which yields a list of values for each point.
        @rtype:  generator
        """
        _DATATYPES = {}

        _DATATYPES[1] = ('b', 1)  # _DATATYPES[PointField.INT8]    = ('b', 1)
        _DATATYPES[2] = ('B', 1)  # _DATATYPES[PointField.UINT8]   = ('B', 1)
        _DATATYPES[3] = ('h', 2)  # _DATATYPES[PointField.INT16]   = ('h', 2)
        _DATATYPES[4] = ('H', 2)  # _DATATYPES[PointField.UINT16]  = ('H', 2)
        _DATATYPES[5] = ('i', 4)  # _DATATYPES[PointField.INT32]   = ('i', 4)
        _DATATYPES[6] = ('I', 4)  # _DATATYPES[PointField.UINT32]  = ('I', 4)
        _DATATYPES[7] = ('f', 4)  # _DATATYPES[PointField.FLOAT32] = ('f', 4)
        _DATATYPES[8] = ('d', 8)  # _DATATYPES[PointField.FLOAT64] = ('d', 8)

        def get_struct_fmt(is_bigendian, fields, field_names=None):
            fmt = ">" if is_bigendian else "<"

            offset = 0
            for field in (
                f
                for f in sorted(fields, key=lambda f: f["offset"])
                if field_names is None or f["name"] in field_names
            ):
                if offset < field["offset"]:
                    fmt += "x" * (field["offset"] - offset)
                    offset = field["offset"]
                if field["datatype"] not in _DATATYPES:
                    print(
                        "Skipping unknown PointField datatype [%d]" % field["datatype"],
                        file=sys.stderr,
                    )
                else:
                    datatype_fmt, datatype_length = _DATATYPES[field["datatype"]]
                    fmt += field["count"] * datatype_fmt
                    offset += field["count"] * datatype_length

            return fmt

        fmt = get_struct_fmt(cloud["is_bigendian"], cloud["fields"], field_names)
        width, height, point_step, row_step, data, isnan = (
            cloud["width"],
            cloud["height"],
            cloud["point_step"],
            cloud["row_step"],
            cloud["data"],
            math.isnan,
        )
        data = base64.b64decode(data)
        data = np.frombuffer(data, dtype=np.uint8)
        unpack_from = struct.Struct(fmt).unpack_from

        if skip_nans:
            if uvs:
                for u, v in uvs:
                    p = unpack_from(data, (row_step * v) + (point_step * u))
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
            else:
                for v in range(height):
                    offset = row_step * v
                    for u in range(width):
                        p = unpack_from(data, offset)
                        has_nan = False
                        for pv in p:
                            if isnan(pv):
                                has_nan = True
                                break
                        if not has_nan:
                            yield p
                        offset += point_step
        else:
            if uvs:
                for u, v in uvs:
                    yield unpack_from(data, (row_step * v) + (point_step * u))
            else:
                for v in range(height):
                    offset = row_step * v
                    for u in range(width):
                        yield unpack_from(data, offset)
                        offset += point_step

    def __estimate_depth_IGEV(self):
        '''
        Estimate depth using IGEV generated disparity map.
        '''
        disparity = self.__estimate_disp_IGEV()
        unified_matrix = self.get_rgb_intrinsics()
        args = self.__get_default_IGEV_args()
        camera_seperation = args.camera_seperation

        # camera intrinsics are for 1080p, so we need to scale them to the current image size
        focal_length = unified_matrix[0][0] * disparity.shape[0] / 1080

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
        script_directory = os.path.dirname(os.path.abspath(sys.argv[0])) 
        model.load_state_dict(torch.load(os.path.join(script_directory, args.restore_ckpt)))
        model = model.module
        model.to('cuda')
        model.eval()

        with torch.no_grad():
            image1 = left_image
            image1_np = np.array(image1[:,:,0:3])
            image2 = right_image
            image2_np = np.array(image2[:,:,0:3])

            # downsampel to 720p to speed up
            image1_np = cv2.resize(image1_np,(1280,720))
            image2_np = cv2.resize(image2_np,(1280,720))

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
        
        # camera intrinsics are for 1080p, so we need to scale them to the current image size        
        cameraMatrix = self.get_rgb_intrinsics() * igev_disp.shape[0] / 1080
        distCoeffs = self.get_rgb_distortion()
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
        # stereo_image = self.__get_image_from_rostopic('/stereo/image_rect_color')
        stereo_image = self.__get_compressedimage_from_rostopic('/stereo/image_rect_color/compressed')        

        # GET RIGHT AND LEFT IMAGES
        left_image = stereo_image[:, :int(stereo_image.shape[1]/2)]
        right_image = stereo_image[:, int(stereo_image.shape[1]/2):]
        
        if stereo_image is None:
            raise Exception('Stereo image not found. Please check the ROS topic name.')

        return left_image, right_image

    def __get_default_IGEV_args(self):
        # parser = argparse.ArgumentParser()
        # # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
        # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='/home/mfi/repos/rtc_vision_toolbox/camera/zed_ros/IGEV/IGEV-Stereo/pretrained_models/middlebury/middlebury.pth')
        # # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/eth3d/eth3d.pth')

        # parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

        # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
        # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.png")

        # parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
        # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        # parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

        # # Architecture choices
        # parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        # parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        # parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
        # parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
        # parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        # parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        # parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        # parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
        # parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

        # parser.add_argument('--left_topic', type=str, default="/zedB/zed_node_B/left/image_rect_color", help="left cam topic")
        # parser.add_argument('--right_topic', type=str, default="/zedB/zed_node_B/right/image_rect_color", help="right cam topic")
        # parser.add_argument('--depth_topic', type=str, default="/zedB/zed_node_B/depth/depth_registered", help="depth cam topic")
        # parser.add_argument('--conf_map_topic', type=str, default="/zedB/zed_node_B/confidence/confidence_map", help="depth confidence map topic")

        # # parser.add_argument('--downsampling', type=bool, default=False, help="downsampling image dimension")
        # parser.add_argument('--downsampling', type=bool, default=False, help="downsampling image dimension")

        # if(self.__camera_type == 'zedx'):
        #     parser.add_argument('--camera_seperation', type=float, default=0.12, help="camera seperation")
        # elif(self.__camera_type == 'zedxm'):
        #     parser.add_argument('--camera_seperation', type=float, default=0.050, help="camera seperation")

        # args = parser.parse_args()
        
        class args:
            restore_ckpt = 'IGEV/IGEV-Stereo/pretrained_models/middlebury/middlebury.pth'  # relative to zed_ros folder
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
            left_topic = "/zedB/zed_node_B/left/image_rect_color"
            right_topic = "/zedB/zed_node_B/right/image_rect_color"
            depth_topic = "/zedB/zed_node_B/depth/depth_registered"
            conf_map_topic = "/zedB/zed_node_B/confidence/confidence_map"
            downsampling = False
            camera_seperation = 0.12
            
        if self.__camera_type == 'zedxm':
            args.camera_seperation = 0.050
        elif self.__camera_type == 'zedx':
            args.camera_seperation = 0.12

        return args

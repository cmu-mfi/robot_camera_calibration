import base64
import math
import struct
import sys
import time
from ctypes import POINTER, c_float, c_uint32, cast, pointer
from datetime import datetime
from typing import Any, Optional, Union

import cv2
import numpy as np
import open3d as o3d
import roslibpy

# TODO start: TOPIC NAMES 
# specify names after namespace. 
# E.g. for a camera with namespace '/camera1', if the rostopic is '/camera1/rgb/image_rect_color',
# then the topic name is '/rgb/image_rect_color'
# Specify None if the topic is not available
RGB_COMPRESSED_TOPIC = '/rgb/image_rect_color/compressed'
RGB_TOPIC = '/rgb/image_rect_color'
DEPTH_TOPIC = '/depth/depth_registered'
POINTCLOUD_TOPIC = '/point_cloud/cloud_registered'
RGB_CAMERA_INFO_TOPIC = '/rgb/camera_info'
DEPTH_CAMERA_INFO_TOPIC = '/depth/camera_info'
# TODO end: TOPIC NAMES

class CameraRos:
    """
    This class represents a camera object for capturing RGB, depth and pointcloud data.
    """
    """
    Methods:
        __init__: Initializes the CameraRos object.
        get_rgb_image: Captures and returns an RGB image from the camera.
        get_raw_depth_data: Captures and returns raw depth data from the camera.
        get_depth_image: Captures and returns a raw depth image from the camera.
        get_colormap_depth_image: Captures and returns a depth image with a colormap applied.
        get_point_cloud: Captures and returns a point cloud from the camera.
        get_rgb_intrinsics: Returns the RGB camera matrix.
        get_depth_intrinsics: Returns the depth camera matrix.
        get_rgb_distortion: Returns the RGB camera distortion coefficients.
        get_depth_distortion: Returns the depth camera distortion coefficients.
    """
    def __init__(self, camera_namespace: str, 
                 rosmaster_ip='localhost', 
                 rosmaster_port=9090, 
                 depth_mode='default',
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

        self.__camera_node = camera_namespace
        self.__depth_mode = depth_mode
    
    def get_rgb_image(self, use_new_frame: bool=True) -> Union[Optional[np.array], Any]:
        """
        Captures and returns an RGB image from the camera.
        """
        
        if RGB_COMPRESSED_TOPIC is not None:
            color_image = self.__get_compressedimage_from_rostopic(RGB_COMPRESSED_TOPIC)
        elif RGB_TOPIC is not None:
            color_image = self.__get_image_from_rostopic(RGB_TOPIC)
        else:
            raise Exception('RGB image topic is not available.')
        
        color_image = np.array(color_image[:,:,0:3])

        return color_image

    def get_raw_depth_data(self, max_depth=None, use_new_frame: bool=True, method=None):
        """
        Captures and returns a raw depth image from the camera.

        Args:
            max_depth (int): Maximum depth value to include in the colormap in mm
        
        Returns:
            depth_data (numpy.ndarray): Raw depth data (in mm) captured from the camera.
        """
        depth_data :np.array = None

        if method is None:
            method = self.__depth_mode

        if method == 'default':
            depth_data = self.__get_image_from_rostopic(DEPTH_TOPIC)
            # convert to mm
            depth_data = depth_data.astype(np.float32) * 1000

        # replace +-inf values with nan
        depth_data = depth_data.copy()
        depth_data[depth_data == float('inf')] = np.nan
        depth_data[depth_data == float('-inf')] = np.nan

        # TODO start: convert meters to mm, if needed
        ...
        # TODO end: convert meters to mm, if needed
            
        if max_depth is not None:
            depth_data = np.where(depth_data < max_depth, depth_data, None)

        return depth_data

    def get_depth_image(self, max_depth = None, use_new_frame: bool=True, method=None):
        """
        Captures and returns a raw depth image from the camera.

        Args:
            max_depth (int): Maximum depth value to include in the colormap in mm
        
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
            point_cloud = self.__get_pointcloud_from_rostopic(POINTCLOUD_TOPIC)

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

        if RGB_CAMERA_INFO_TOPIC is None:
            raise Exception('RGB camera info topic is not available.')
        
        camera_info_sub = roslibpy.Topic(self.__ros_client, '/'+self.__camera_node + RGB_CAMERA_INFO_TOPIC, 'sensor_msgs/CameraInfo')        

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
        
        if DEPTH_CAMERA_INFO_TOPIC is None:
            raise Exception('Depth camera info topic is not available.')
        
        camera_info_sub = roslibpy.Topic(self.__ros_client, '/'+self.__camera_node + DEPTH_CAMERA_INFO_TOPIC, 'sensor_msgs/CameraInfo')        

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
        
        if RGB_CAMERA_INFO_TOPIC is None:
            raise Exception('RGB camera info topic is not available.')
        
        camera_info_sub = roslibpy.Topic(self.__ros_client, '/'+self.__camera_node + RGB_CAMERA_INFO_TOPIC, 'sensor_msgs/CameraInfo')        

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
        
        if DEPTH_CAMERA_INFO_TOPIC is None:
            raise Exception('Depth camera info topic is not available.')
        
        camera_info_sub = roslibpy.Topic(self.__ros_client, '/'+self.__camera_node + DEPTH_CAMERA_INFO_TOPIC, 'sensor_msgs/CameraInfo')        

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
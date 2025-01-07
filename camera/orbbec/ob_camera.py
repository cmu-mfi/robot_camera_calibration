from pyorbbecsdk import *
import cv2
import numpy as np
import open3d as o3d
import time
from datetime import datetime

from typing import Union, Any, Optional

class OBCamera:
    """
    This class represents a camera object for capturing RGB, depth, IR images and pointcloud data.
    """
    """
    Methods:
        __init__(serial_no=None): Initializes the Camera object.
        get_rgb_image(): Captures and returns an RGB image from the camera.
        get_ir_image(): Captures and returns an IR image from the camera.
        get_raw_depth_image(): Captures and returns a raw depth image from the camera.
        get_colormap_depth_image(min_depth=20, max_depth=10000, colormap=cv2.COLORMAP_JET): Captures and returns a depth image with a colormap applied.
        get_point_cloud(min_mm: int = 400, max_mm: int = 1000, save_points=False): Captures and returns a point cloud from the camera.
        get_rgb_intrinsics(): Returns the camera matrix.
        get_depth_intrinsics(): Returns the depth camera matrix.
        get_rgb_distortion(): Returns the rgb camera distortion coefficients.
        get_depth_distortion(): Returns the depth camera distortion coefficients. 
    
    Troubleshooting:

        Using multiple cameras (e.g. nvram error) https://github.com/OpenKinect/libfreenect2/issues/807
    """

    # data members
    __pipeline: Pipeline = None  # Pipeline object for capturing frames
    __config: Config = None  # Config object for configuring the camera
    camera_param: OBCameraParam = None # Camera parameters
    __sn: str = None  # Serial number of the camera
    __device: Device = None  # Device object for the camera
    __latest_frameset: FrameSet = None # Latest frameset object
    
    def __init__(self, device_index: int=0, serial_no: str=None):
        """
        Initializes the Camera object.

        Args:
            device_index (int): Index of the camera device. Default is 0.
            serial_no (str): Serial number of the camera. If None, uses device_index.
        """

        # Choose the first camera if serial_no is None
        ctx = Context()
        ctx.set_logger_level(OBLogLevel.ERROR)
        device_list = ctx.query_devices()
        if serial_no is None:
            if len(device_list) == 0:
                print("No device connected")
                return
            self.__device = device_list.get_device_by_index(device_index)
        else:
            self.__device = device_list.get_device_by_serial_number(serial_no)

        device_info = self.__device.get_device_info()
        self.__sn = device_info.get_serial_number()
        
        self.__pipeline = Pipeline(self.__device)
        time.sleep(2) # wait for the camera to initialize
        self.__config = Config()
        self.camera_param = self.__pipeline.get_camera_param()
        
        # Initialize color stream profile
        profile_list = self.__pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1280, 0, OBFormat.RGB, 15)
        # color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
        self.__config.enable_stream(color_profile)

        # Initialize depth stream profile
        profile_list = self.__pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1024, 0, OBFormat.Y16, 15)
        # depth_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
        self.__config.enable_stream(depth_profile)

        # Initialize ir stream profile
        profile_list = self.__pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
        ir_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1024, 0, OBFormat.Y16, 15)
        # ir_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
        self.__config.enable_stream(ir_profile)
        
        # Align the color and depth streams
        device_pid = device_info.get_pid()
        try:
            if device_pid == 0x066B:
                #Femto Mega does not support hardware D2C, and it is changed to software D2C
                self.__config.set_align_mode(OBAlignMode.SW_MODE)
            else:
                self.__config.set_align_mode(OBAlignMode.HW_MODE)
        except OBError as e:
            self.__config.set_align_mode(OBAlignMode.DISABLE)
            print(e)
        try:
            self.__pipeline.enable_frame_sync()
        except Exception as e:
            print(e)
        
        self.__pipeline.start(self.__config)
        self.camera_param = self.__pipeline.get_camera_param()
        time.sleep(2) # wait for the camera to start
        print("Camera initialized: ", self.__sn)
        print("Camera Params: ", self.camera_param)
        self.get_frame()
        
    def __del__(self):
        """
        Stops the camera pipeline when the Camera object is deleted.
        """
        self.__pipeline.stop()

    def get_frame(self):
        """
        Returns the current frameset object.

        Returns:
            frameset: Current frameset object.
        """
        frames = self.__pipeline.wait_for_frames(100)
        if frames is None:
            print("failed to get frames")
            return None
        
        self.__latest_frameset = frames
        return frames

    def get_rgb_image(self, use_new_frame: bool=True) -> Union[Optional[np.array], Any]:
        """
        Captures and returns an RGB image from the camera.

        Args:
            use_new_frame (bool): If True, captures a new frame. If False, uses the latest frame.
        
        Returns:
            color_image (numpy.ndarray): RGB image captured from the camera.
        """

        if use_new_frame:
            frames = self.get_frame()
        else:
            frames = self.__latest_frameset

        if frames is None:
            print("failed to get frames")
            return None
        color_frame: VideoFrame = frames.get_color_frame()
        if color_frame is None:
            print("failed to get color frame")
            return None

        # covert to frame to an image
        color_image = self.__frame_to_bgr_image(color_frame)
        if color_image is None:
            print("failed to convert frame to image")
            return None
        
        return color_image
    
    def get_ir_image(self, use_new_frame: bool=True):
        """
        Captures and returns an IR image from the camera.

        Args:
            use_new_frame (bool): If True, captures a new frame. If False, uses the latest frame.
        
        Returns:
            ir_image (numpy.ndarray): IR image captured from the camera.
        """
        if use_new_frame:
            frames = self.get_frame()
        else:
            frames = self.__latest_frameset

        if frames is None:
            print("failed to get frames")
            return None
        ir_frame: VideoFrame = frames.get_ir_frame()
        if ir_frame is None:
            print("failed to get ir frame")
            return None
        
        ir_data = np.asanyarray(ir_frame.get_data())
        width = ir_frame.get_width()
        height = ir_frame.get_height()
        ir_format = ir_frame.get_format()
        if ir_format == OBFormat.Y8:
            ir_data = np.resize(ir_data, (height, width, 1))
            data_type = np.uint8
            image_dtype = cv2.CV_8UC1
            max_data = 255
        elif ir_format == OBFormat.MJPG:
            ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
            data_type = np.uint8
            image_dtype = cv2.CV_8UC1
            max_data = 255
            if ir_data is None:
                print("decode mjpeg failed")
            else:
                ir_data = np.resize(ir_data, (height, width, 1))
        else:
            ir_data = np.frombuffer(ir_data, dtype=np.uint16)
            data_type = np.uint16
            image_dtype = cv2.CV_16UC1
            max_data = 65535
            ir_data = np.resize(ir_data, (height, width, 1))
        cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
        ir_data = ir_data.astype(data_type)
        ir_image = cv2.cvtColor(ir_data, cv2.COLOR_GRAY2RGB)

        return ir_image

    def get_raw_depth_data(self, max_depth = None, use_new_frame: bool=True):
        """
        Captures and returns a raw depth image from the camera.

        Args:
            max_depth (int): Maximum depth value to include in the depth data in mm.
            use_new_frame (bool): If True, captures a new frame. If False, uses the latest frame.
        
        Returns:
            depth_data (numpy.ndarray): Raw depth data (in mm) captured from the camera.
        """
        if use_new_frame:
            frames = self.get_frame()
        else:
            frames = self.__latest_frameset
        
        if frames is None:
            print("failed to get frames")
            return None
        depth_frame: VideoFrame = frames.get_depth_frame()
        if depth_frame is None:
            print("failed to get depth frame")
            return None

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))
        depth_data = depth_data.astype(np.float32) * scale
        
        if max_depth is not None:
            depth_data = np.where(depth_data < max_depth, depth_data, None)

        return depth_data
    
    def get_depth_image(self, max_depth = None, use_new_frame: bool=True):
        """
        Captures and returns a raw depth image from the camera.

        Args:
            use_new_frame (bool): If True, captures a new frame. If False, uses the latest frame.
        
        Returns:
            depth_image (numpy.ndarray): Raw depth image captured from the camera.
        """
        if use_new_frame:
            frames = self.get_frame()
        else:
            frames = self.__latest_frameset
        
        if frames is None:
            print("failed to get frames")
            return None
        depth_frame: VideoFrame = frames.get_depth_frame()
        if depth_frame is None:
            print("failed to get depth frame")
            return None

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))
        depth_data = depth_data.astype(np.float32) * scale
        
        if max_depth is not None:
            depth_data = np.where(depth_data < max_depth, depth_data, None)

        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)
        
        return depth_image

    def get_colormap_depth_image(self, min_depth: int=20, max_depth: int=10000,
                                 colormap: int=cv2.COLORMAP_JET, use_new_frame: bool=True):
        """
        Captures and returns a depth image with a colormap applied.

        Args:
            min_depth (int): Minimum depth value to include in the colormap.
            max_depth (int): Maximum depth value to include in the colormap.
            colormap (int): OpenCV colormap to apply to the depth image.
            use_new_frame (bool): If True, captures a new frame. If False, uses the latest frame.

        Returns:
            depth_image (numpy.ndarray): Depth image with colormap applied.
        """
        depth_image = self.get_depth_image(use_new_frame=use_new_frame)
        depth_image = np.where((depth_image > min_depth) & (depth_image < max_depth),
                               depth_image, 0)
        depth_image = cv2.applyColorMap(depth_image, colormap)

        return depth_image
    
    def get_point_cloud(self, min_mm: int = 10, max_mm: int = 10000, 
                        save_points: bool=False, use_new_frame: bool=True) -> o3d.geometry.PointCloud:
        """
        Captures and returns a point cloud from the camera.

        Returns:
            point_cloud (o3d.geometry.PointCloud): Point cloud captured from the camera.
        """
        if use_new_frame:
            frames = self.get_frame()
        else:
            frames = self.__latest_frameset
        
        if frames is None:
            print("failed to get frames")
            return None
        depth_frame: VideoFrame = frames.get_depth_frame()
        if depth_frame is None:
            print("failed to get depth frame")
            return None
        points = frames.get_point_cloud(self.camera_param)
        if points is None or len(points) == 0:
            print("Points is None or is empty")
            return None
        
        points = np.array(points)
        print(f'Points shape: {points.shape}')
        new_points = []
        for point in points:
            if point[-1] < max_mm and point[-1] > min_mm:
                new_points.append(point)
        points = np.array(new_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if(save_points):
            now = datetime.now().strftime("%m%d_%H%M")
            points_filename = f"points_{now}.ply"
            # points_filename = f"points_{depth_frame.get_timestamp()}.ply"
            o3d.io.write_point_cloud(points_filename, pcd)
        
        return pcd

    def get_rgb_intrinsics(self):
        """
        Returns the camera matrix.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]

        Returns:
            camera_matrix: Intrinsics of the camera as camera matrix.
        """
        self.camera_param = self.__pipeline.get_camera_param()
        rgbIntrinsics = self.camera_param.rgb_intrinsic
        camera_matrix = np.array([[rgbIntrinsics.fx, 0, rgbIntrinsics.cx],
                                  [0, rgbIntrinsics.fy, rgbIntrinsics.cy],
                                  [0, 0, 1]])
        
        return camera_matrix
    
    def get_depth_intrinsics(self):
        """
        Returns the depth camera matrix.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]

        Returns:
            camera_matrix: Intrinsics of the camera as camera matrix.
        """
        self.camera_param = self.__pipeline.get_camera_param()
        depthIntrinsics = self.camera_param.depth_intrinsic
        
        camera_matrix = np.array([[depthIntrinsics.fx, 0, depthIntrinsics.cx],
                                  [0, depthIntrinsics.fy, depthIntrinsics.cy],
                                  [0, 0, 1]])
        
        return camera_matrix
    
    def get_rgb_distortion(self):
        """
        Returns the rgb camera distortion coefficients.

        Returns:
            distortion: Distortion coefficients of the camera.
        """
        self.camera_param = self.__pipeline.get_camera_param()
        rgbDistortion = self.camera_param.rgb_distortion
        distortion = np.array([rgbDistortion.k1, rgbDistortion.k2, rgbDistortion.p1, rgbDistortion.p2, rgbDistortion.k3])
        
        return distortion
    
    def get_depth_distortion(self):
        """
        Returns the depth camera distortion coefficients.

        Returns:
            distortion: Distortion coefficients of the camera.
        """
        self.camera_param = self.__pipeline.get_camera_param()
        depthDistortion = self.camera_param.depth_distortion
        distortion = np.array([depthDistortion.k1, depthDistortion.k2, depthDistortion.p1, depthDistortion.p2, depthDistortion.k3])
        
        return distortion
    
    def __frame_to_bgr_image(self, frame: VideoFrame):
        """
        Convert a frame to a BGR image.

        Args:
            frame (VideoFrame): The frame to convert.

        Returns:
            The BGR image converted from the frame.

        Raises:
            None.

        """
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        image = np.zeros((height, width, 3), dtype=np.uint8)
        if color_format == OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_format == OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        elif color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif color_format == OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        else:
            print("Unsupported color format: {}".format(color_format))
            return None
        return image

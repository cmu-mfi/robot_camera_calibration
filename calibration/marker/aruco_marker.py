import cv2
import cv2.aruco as aruco
import numpy as np
from autolab_core import Point, CameraIntrinsics, RigidTransform

class ArucoMarker:
    """
    Class representing an Aruco marker for camera calibration.
    """
    
    """
    Data Members:
    - __size (float): Size of the marker in meters.
    - __detector (ArucoDetector): Aruco detector object.
    
    Methods:
    - __init__(type='DICT_4X4_100', size=0.1): Constructor for the ArucoMarker class.
    - get_center_poses(input_image, camera_matrix=None, camera_distortion=None): Detects Aruco markers in the input image and estimates their center poses.
    - __detect_markers(input_image): Detects Aruco markers in the input image.
    - __estimatePoseSingleMarkers(corners, marker_size, mtx, distortion): Estimates the pose of single Aruco markers.
    """

    def __init__(self, type='DICT_4X4_100', size=0.1, debug=False):
        """
        Constructor for the ArucoMarker class.

        Parameters:
        - type (str): Aruco dictionary type. Default is 'DICT_4X4_100'.
        - size (float): Size of the marker in meters. Default is 0.1.
        """
        self.__size = size
        self.debug = debug
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, type))
        parameters = aruco.DetectorParameters()
        self.__detector = aruco.ArucoDetector(dictionary, parameters)

    def get_center_poses(self, input_image, camera_matrix=None, camera_distortion=None, depth_image=None, debug=False):
        """
        Detects Aruco markers in the input image and estimates their center poses.

        Parameters:
        - input_image (numpy.ndarray): Input image containing Aruco markers.
        - camera_matrix (numpy.ndarray): Camera matrix. Default is None.
        - camera_distortion (numpy.ndarray): Camera distortion coefficients. Default is None.
        - depth_image (numpy.ndarray): Depth image. Default is None.

        Returns:
        - Transforms (list): List of transformation matrices representing the center poses of the detected markers.
        - ids (numpy.ndarray): Array of marker IDs.
        """
        corners, ids, rejected = self.__detect_markers(input_image)
        
        if debug or self.debug:
            print("Detected markers: ", ids)
            output_image = input_image.copy()
            cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
            #cv2.drawFrameAxes(output_image, camera_matrix, camera_distortion, rvecs, tvecs, self.__size)
            cv2.imwrite("debug.png", output_image)
        
        if(ids is None):
            return None, None

        if camera_matrix is None:
            camera_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        if camera_distortion is None:
            camera_distortion = np.zeros(5)

        rvecs, tvecs, _ = self.__estimatePoseSingleMarkers(corners, self.__size, camera_matrix, camera_distortion)

            
        Transforms = []
        
        for i in range(len(ids)):
            rvec, tvec = rvecs[i], tvecs[i]
            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            # Form the transformation matrix
            transform_mat = np.eye(4)
            transform_mat[:3, :3] = rmat
            transform_mat[:3, 3] = tvec.squeeze()
            Transforms.append(transform_mat)
            
        if depth_image is not None:
            Transforms = self.__improve_marker_poses(Transforms, corners, depth_image, camera_matrix)
            
        return Transforms, ids

    def __improve_marker_poses(self, Transforms, corners, depth_image, camera_matrix):
        """
        Improves the marker poses by using depth information.

        Parameters:
        - Transforms (list): List of transformation matrices representing the center poses of the detected markers.
        - corners (list): List of marker corners.
        - depth_image (numpy.ndarray): Depth image.
        - camera_matrix (numpy.ndarray): Camera matrix.

        Returns:
        - Transforms (list): List of improved transformation matrices.
        """
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        intrinsics = CameraIntrinsics('cam',fx, fy, cx, cy)
             
        for i in range(len(Transforms)):
            transform_mat = Transforms[i]
            
            corner = corners[i].squeeze()
            corner = corner.reshape((4,2)).astype(int)
            print(corner)

            image_x = corner[:, 0].mean()
            image_y = corner[:, 1].mean()
            
            object_center = Point(np.array([image_x, image_y]), 'cam')
            object_depth = np.mean(depth_image[int(image_y)-1:int(image_y)+1, int(image_x)-1:int(image_x)+1])
        
            object_center = intrinsics.deproject_pixel(depth = object_depth, 
                                                       pixel = object_center)
            
            print("Object center: ", object_center)
            print("Object depth: ", object_depth)
            
            # Update the translation vector
            transform_mat[:3, 3] = object_center.data
            Transforms[i] = transform_mat
        
        return Transforms
            

    def __detect_markers(self, input_image):
        """
        Detects Aruco markers in the input image.

        Parameters:
        - input_image (numpy.ndarray): Input image containing Aruco markers.

        Returns:
        - corners (list): List of marker corners.
        - ids (numpy.ndarray): Array of marker IDs.
        - rejected_img_points (list): List of rejected marker corners.
        """
        corners, ids, rejected_img_points = self.__detector.detectMarkers(input_image)
        
        if ids is None:
            print("No markers found.")
        else:
            print("Detected markers: ", ids)
        
        return corners, ids, rejected_img_points

    def __estimatePoseSingleMarkers(self, corners, marker_size, mtx, distortion):
        """
        Estimates the pose of single Aruco markers.

        Parameters:
        - corners (list): List of marker corners.
        - marker_size (float): Size of the marker in meters.
        - mtx (numpy.ndarray): Camera matrix.
        - distortion (numpy.ndarray): Camera distortion coefficients.

        Returns:
        - rvecs (list): List of rotation vectors.
        - tvecs (list): List of translation vectors.
        - trash (list): List of trash values.
        """
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        trash = []
        rvecs = []
        tvecs = []
        
        for c in corners:
            nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)
            trash.append(nada)
        return rvecs, tvecs, trash
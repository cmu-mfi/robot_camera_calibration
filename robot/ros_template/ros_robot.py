import numpy as np
from scipy.spatial.transform import Rotation as R

import roslibpy
# https://github.com/gramaziokohler/roslibpy
# depends on rosbridge_server rosbridge_websocket running on the ROS side

import roslibpy.tf as tf
# depends on tf2_web_publisher (http://wiki.ros.org/tf2_web_republisher) on the ROS side

class ROSRobot:
    
    def __init__(self, robot_namespace='', rosmaster_ip='localhost', rosmaster_port=9090):
        
        # initialize ros client
        self.ros_client = roslibpy.Ros(host=rosmaster_ip, port=rosmaster_port)
        self.ros_client.run()
        print('Is ROS connected?', self.ros_client.is_connected)      
        
        self.__robot_name = robot_namespace
        
        # subscribe to rosservices
        # TODO start
        # Add necessary services for the robot required for moving the robot
        # Example 1: self.__move_to_pose_service = roslibpy.Service(self.ros_client, '/'+robot_namespace+'/yk_set_pose', 'yk_tasks/SetPose')
        # Example 2: self.__enable_robot_service = roslibpy.Service(self.ros_client, '/'+robot_namespace+'/robot_enable', 'std_srvs/Trigger')
        
        ...
        
        # TODO ends
    
    def get_eef_pose(self):
        '''
        Get the latest end effector pose
        The method uses ROS tf client to get the latest transformation data. 
        If ROS tf tree doesn't have the required frame, implement the method from keeping arguments and return variables same.
        
        Arguments: None
        Returns: np.ndarray(4x4)        
        '''
        
        tf_client   = tf.TFClient(self.ros_client, 
                        fixed_frame = '/' + self.__robot_name + '/base_link', 
                        angular_threshold=2.0, 
                        translation_threshold=0.01, 
                        rate=10.0, 
                        server_name='/tf2_web_republisher')
        
        self.latest_eef_pose = None
        
        # TODO starts: Uncomment below line and replace /tool0 with the correct end effector frame name
        # tf_client.subscribe('/' + self.__robot_name + '/tool0', self.__tf_client_callback)
        
        ...
        
        # TODO ends
        
        while self.latest_eef_pose is None:
            pass       

        return self.latest_eef_pose
    
    def move_to_pose(self, position, orientation, max_velocity_scaling_factor=0.3, max_acceleration_scaling_factor=0.3):
        
        position = np.array(position)
        orientation = np.array(orientation)
        
        # validate if orientation is a quarternion (4x1) or a rotation matrix (3x3)
        if orientation.shape == (3,3):
            rotation = R.from_matrix(orientation)
            orientation = rotation.as_quat()
        elif orientation.shape == (4,):
            pass
        else:
            return False

        # TODO starts: Implement interface to move the robot to a given pose
        
        ...
        
        # TODO ends           
                
    def __tf_client_callback(self, data):
        '''
        Callback function for tf client
        
        Parameters:
        data : dict
            dictionary containing the latest transformation data
            Example: {'translation': {'x': -0.16239415521463996, 'y': -0.25082783397043434, 'z': 0.6049835542505466}, 
                     'rotation': {'x': -0.5755621019981999, 'y': -0.4107506164843213, 'z': 0.410702435616227, 'w': 0.5756176744854713}}
        '''
        #convert the data to a transformation 4x4 matrix
        
        rotation = R.from_quat([data['rotation']['x'], data['rotation']['y'], data['rotation']['z'], data['rotation']['w']])
        rotation = rotation.as_matrix().astype(np.float64)
        translation = np.array([data['translation']['x'], data['translation']['y'], data['translation']['z']]).astype(np.float64)
        
        self.latest_eef_pose = np.eye(4)
        self.latest_eef_pose[:3,:3] = rotation
        self.latest_eef_pose[:3,3] = translation
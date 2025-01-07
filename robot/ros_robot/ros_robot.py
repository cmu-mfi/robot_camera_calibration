import numpy as np
from scipy.spatial.transform import Rotation as R

import roslibpy
# https://github.com/gramaziokohler/roslibpy
# depends on rosbridge_server rosbridge_websocket running on the ROS side

import roslibpy.tf as tf
# depends on tf2_web_publisher (http://wiki.ros.org/tf2_web_republisher) on the ROS side

class ROSRobot:

    #data members
    ros_client = None
    __move_to_pose_service = None
    __robot_name = None
    latest_eef_pose = None #dict containing 'translation' and 'rotation' keys
    
    #constructor    
    def __init__(self, robot_name='yk_builder', rosmaster_ip='localhost', rosmaster_port=9090):
        
        # initialize ros client
        self.ros_client = roslibpy.Ros(host=rosmaster_ip, port=rosmaster_port)
        self.ros_client.run()
        print('Is ROS connected?', self.ros_client.is_connected)      
        
        self.__robot_name = robot_name
        
        # subscribe to rosservices
        self.__move_to_pose_service = roslibpy.Service(self.ros_client, '/'+robot_name+'/yk_set_pose', 'yk_tasks/SetPose')
        self.__move_to_joints_service = roslibpy.Service(self.ros_client, '/'+robot_name+'/yk_set_joints', 'yk_tasks/SetJoints')      
        self.__enable_robot_service = roslibpy.Service(self.ros_client, '/'+robot_name+'/robot_enable', 'std_srvs/Trigger')  
    
    def get_eef_pose(self):
        # initialize tf client
        tf_client   = tf.TFClient(self.ros_client, 
                        fixed_frame = '/' + self.__robot_name + '/base_link', 
                        angular_threshold=2.0, 
                        translation_threshold=0.01, 
                        rate=10.0, 
                        server_name='/tf2_web_republisher')
        
        self.latest_eef_pose = None
        
        tf_client.subscribe('/' + self.__robot_name + '/tool0', self.__tf_client_callback)
        
        while self.latest_eef_pose is None:
            pass       
        
        #print('Latest EEF Pose:\n', self.latest_eef_pose)
       
        return self.latest_eef_pose
    
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

        #serialize pose and send to the service
        pose_serial = {'position': {'x': position[0], 'y': position[1], 'z': position[2]},
                       'orientation': {'x': orientation[0], 'y': orientation[1], 'z': orientation[2], 'w': orientation[3]}}
        try:
            enable_request = roslibpy.ServiceRequest({})
            enable_response = self.__enable_robot_service.call(enable_request)
            request = roslibpy.ServiceRequest({'pose': pose_serial, 
                                               'base_frame': 'base_link',
                                               'max_velocity_scaling_factor': max_velocity_scaling_factor,
                                               'max_acceleration_scaling_factor': max_acceleration_scaling_factor})
            response = self.__move_to_pose_service.call(request)            
            return True
        
        except Exception as e:
            print('Try again. Error:', e)
            return False
    
    def move_to_joints(self, joint_values):
        
        joint_values = [joint_values[i] for i in range(len(joint_values))]
        joint_names = ['joint_1_s', 'joint_2_l', 'joint_3_u', 'joint_4_r', 'joint_5_b', 'joint_6_t']
        
        #serialize message and send to the service
        joint_state_serial = {'name': list(joint_names), 'position': joint_values}
        
        try:
            enable_request = roslibpy.ServiceRequest({})
            enable_response = self.__enable_robot_service.call(enable_request)
            request = roslibpy.ServiceRequest({'state': joint_state_serial})
            response = self.__move_to_joints_service.call(request)
            return True
        
        except Exception as e:
            print('Try again. Error:', e)
            return False
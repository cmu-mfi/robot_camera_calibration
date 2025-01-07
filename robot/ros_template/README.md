## Pre-requisites

* The class interfaces with ROS via a [ROSBridge websocket server](https://github.com/RobotWebTools/rosbridge_suite).

* Install and run the ROSBridge server on your ROS machine by following instructions at \
[https://github.com/cmu-mfi/rosbridge_suite.git](https://github.com/cmu-mfi/rosbridge_suite.git)


## TODOs `ros_robot.py`

1. `__init__`: Add necessary services clients for the robot required for moving the robot
2. `move_to_pose`: Implement interface to move the robot to a given pose
3. `get_eef_pose`: Implement interface to get the latest end effector pose

## Testing the class

1. Create a venv and install the necessary requirements
2. Run the ROSBridge server on your ROS machine
3. Run the `tests/test.py` script to test the class
    * test1: Console output of the current pose of the robot
    * test2: The robot should move randomly to a pose near current pose
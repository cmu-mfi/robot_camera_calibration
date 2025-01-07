## Pre-requisites

* The class interfaces with ROS via a [ROSBridge websocket server](https://github.com/RobotWebTools/rosbridge_suite).

* Install and run the ROSBridge server on your ROS machine by following instructions at \
[https://github.com/cmu-mfi/rosbridge_suite.git](https://github.com/cmu-mfi/rosbridge_suite.git)


## TODOs `camera_ros.py`

1. TOPIC NAMES: Change the topic names in the `camera_ros.py` to match the topics in your ROS environment
2. `get_raw_depth_data` function: Add a meteres to millimeters conversion factor if necessary

## Testing the class

1. Create a venv and install the necessary requirements
2. Run the ROSBridge server on your ROS machine
3. Run the `tests/test.py` script to test the class
    1. **Get RGB/DEPTH intrinsics**: Retrieve the intrinsic parameters for the RGB and depth cameras.
    2. **Get RGB image**: Capture and return an RGB image from the camera.
    3. **Get default depth image**: Capture and return a depth image from the camera.
    4. **Get default point cloud**: Generate and return a point cloud from the depth data.

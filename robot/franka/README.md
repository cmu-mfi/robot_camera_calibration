## Pre-requisites

* The class interfaces with ROS via [Frankapy](https://github.com/iamlab-cmu/frankapy).
* Follow the setup instructions detailed in that repo.
* For more details and troubleshooting, checkout this [Frankapy fork](https://github.com/jacintosuner/frankapy).


## Testing the class

1. Create a venv and install the necessary requirements
2. Run the ROSBridge server on your ROS machine
3. Run the `tests/test.py` script to test the class
    * test1: Console output of the current pose of the robot
    * test2: The robot should move randomly to a pose near current pose
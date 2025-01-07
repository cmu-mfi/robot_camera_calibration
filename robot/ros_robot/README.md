## Start Robot ROS Node

```
$ ssh mfi@172.26.179.142
password: lego
$ roslaunch testbed_utils lego_moveit_yk.launch namespace:=yk_builder
```

### Enable Robot
The above launch enables the robot to receive remote commands. 
But, in case the robot is toggled from/to remote mode, enabling robot is required.

*In a new terminal*
```
$ ssh mfi@172.26.179.142
password: lego
$ rosservice call /yk_destroyer/robot_enable
success: True
```
If the terminal output doesn't show `success: True`. Fix the issue using text in `message`, and try the above command again.
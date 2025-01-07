# messages/geometry_msgs.py

class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

class Pose:
    def __init__(self, position=None, orientation=None):
        if position is None:
            position = Point()
        if orientation is None:
            orientation = Quaternion()
        self.position = position
        self.orientation = orientation

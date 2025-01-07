import cv2
import numpy as np

from camera.camera_ros.camera_ros import CameraRos

# TEST1: GET RGB/DEPTH INTRINSICS.
def test1(camera):
    rgb_intrinsics = camera.get_rgb_intrinsics()
    depth_intrinsics = camera.get_depth_intrinsics()
    print("RGB intrinsics: \n", rgb_intrinsics)
    print("Depth intrinsics: \n", depth_intrinsics)


# TEST2: GET RGB IMAGE.
def test2(camera):
    image = camera.get_rgb_image()
    cv2.imwrite("rgb_image.png", image)


# TEST3: GET DEFAULT DEPTH IMAGE and DATA.
def test3(camera):
    depth_image = camera.get_raw_depth_data()

    print("Depth image shape: ", depth_image.shape)
    print("Depth image type: ", depth_image.dtype)
    print("Depth image: \n", depth_image[:4, :4])
    print("Max depth value: ", np.nanmax(depth_image))
    print("Min depth value: ", np.nanmin(depth_image))

    np.save("depth_data.npy", depth_image)

    depth_image = camera.get_depth_image()

    cv2.imwrite("depth_image.png", depth_image)

    depth_image = camera.get_colormap_depth_image()
    cv2.imwrite("depth_image_colormap.png", depth_image)


# TEST4: GET DEFAULT POINT CLOUD.
def test4(camera):
    print("Getting default point cloud")
    point_cloud = camera.get_point_cloud(method='default', save_points=True)
    print("Point cloud shape: ", np.asarray(point_cloud.points).shape)


if __name__ == "__main__":

    print("Testing ZedRos class")

    camera_namespace = input("Enter camera namespace: ")
    
    camera = CameraRos(camera_namespace=camera_namespace)

    print("Available tests: ")
    print("1. Get RGB/DEPTH intrinsics")
    print("2. Get RGB image")
    print("3. Get default depth image")
    print("4. Get default point cloud")
    test = input("Enter test number (1-4): ")

    match test:
        case "1":
            test1(camera)
        case "2":
            test2(camera)
        case "3":
            test3(camera)
        case "4":
            test4(camera)
        case _:
            print("Invalid test number")

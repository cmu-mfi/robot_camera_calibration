import argparse

import cv2
import numpy as np

from camera.zed_ros.zed_ros import ZedRos


# TEST1: GET RGB/DEPTH INTRINSICS. WORKS OK ✓
def test1(camera):
    rgb_intrinsics = camera.get_rgb_intrinsics()
    depth_intrinsics = camera.get_depth_intrinsics()
    print("RGB intrinsics: \n", rgb_intrinsics)
    print("Depth intrinsics: \n", depth_intrinsics)


# TEST2: GET RGB IMAGE. WORKS OK ✓
def test2(camera):
    image = camera.get_rgb_image()
    cv2.imwrite("rgb_image.png", image)


# TEST3: GET DEFAULT DEPTH IMAGE and DATA. WORKS OK ✓
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


# TEST4: GET DEFAULT POINT CLOUD. To be Implemented
def test4(camera):
    print("Getting default point cloud")
    point_cloud = camera.get_point_cloud(method='default', save_points=True)
    print("Point cloud shape: ", np.asarray(point_cloud.points).shape)


# TEST5: GET IGEV DEPTH IMAGE. WORKS OK ✓
def test5(camera):
    depth_image = camera.get_raw_depth_data(method="igev", max_depth=1000)

    print("Depth image shape: ", depth_image.shape)
    print("Depth image type: ", depth_image.dtype)
    print("Depth image: \n", depth_image[:4, :4])
    # print("Max depth value: ", np.nanmax(depth_image))
    # print("Min depth value: ", np.nanmin(depth_image))
    np.save("depth_igev_data.npy", depth_image)

    igev_depth_image = camera.get_depth_image(method="igev")
    cv2.imwrite("igev_depth_image.png", igev_depth_image)
    igev_depth_image = camera.get_colormap_depth_image(method="igev")
    cv2.imwrite("igev_depth_image_colormap.png", igev_depth_image)


# TEST6: GET IGEV POINT CLOUD. WORKS OK ✓
def test6(camera):
    print("Getting point cloud using IGEV method")
    point_cloud = camera.get_point_cloud(method="igev", save_points=True, max_mm = 2000)
    print("Point cloud shape: ", np.asarray(point_cloud.points).shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Test script for ZedRos class")
    parser.add_argument("--camera_node", type=str,default="/cam0/zed_cam0", help="Camera node name")
    parser.add_argument("--camera_type", type=str,default="zedx", help="Camera type (like zedx, zedxm)")
    parser.add_argument("--test", type=int, help="Test number (1-6)")
    return parser.parse_args()

if __name__ == "__main__":

    print("Testing ZedRos class")

    args = parse_args()

    camera = ZedRos(camera_node=args.camera_node, camera_type=args.camera_type)

    if args.test:
        test = args.test
    else:
        print("Suggestion: Use --test option to specify test number")        
        print("Available tests: ")
        print("1. Get RGB/DEPTH intrinsics")
        print("2. Get RGB image")
        print("3. Get default depth image")
        print("4. Get default point cloud")
        print("5. Get IGEV depth image")
        print("6. Get IGEV point cloud")
        test = int(input("Enter test number (1-6): "))

    match test:
        case 1:
            test1(camera)
        case 2:
            test2(camera)
        case 3:
            test3(camera)
        case 4:
            test4(camera)
        case 5:
            test5(camera)
        case 6:
            test6(camera)
        case _:
            print("Invalid test number")

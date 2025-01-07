import argparse

import cv2
import numpy as np

from camera.orbbec.ob_camera import OBCamera

cam_sn = {
    'cam0': 'CL8FC3100RL',
    'cam1': 'CL8FC3100W3',
    'cam2': 'CL8FC3100NM',
    'cam3': 'CL8FC3100R4'
}

# TEST1: GET RGB/DEPTH INTRINSICS. WORKS OK ✓
def test1(camera):        
    rgb_intrinsics = camera.get_rgb_intrinsics()
    depth_intrinsics = camera.get_depth_intrinsics()
    print("RGB intrinsics: \n", rgb_intrinsics)
    print("Depth intrinsics: \n", depth_intrinsics)
    
# TEST2: GET RGB IMAGE. WORKS OK ✓
def test2(camera):
    image = camera.get_rgb_image()
    cv2.imwrite('rgb_image.png', image)

# TEST3: GET DEFAULT DEPTH IMAGE and DATA. WORKS OK ✓
def test3(camera):
    depth_image = camera.get_raw_depth_data()    
    
    print("Depth image shape: ", depth_image.shape)
    print("Depth image type: ", depth_image.dtype)
    print("Depth image: \n", depth_image[:4, :4])
    print("Max depth value: ", np.nanmax(depth_image))
    print("Min depth value: ", np.nanmin(depth_image))
    
    np.save('depth_data.npy', depth_image)

    depth_image = camera.get_depth_image()
    
    cv2.imwrite('depth_image.png', depth_image)
    
    depth_image = camera.get_colormap_depth_image()
    cv2.imwrite('depth_image_colormap.png', depth_image)
    
# TEST4: GET DEFAULT POINT CLOUD.
def test4(camera):
    print("Getting default point cloud")
    point_cloud = camera.get_point_cloud(save_points=True)
    print("Point cloud shape: ", np.asarray(point_cloud.points).shape)
    
# TEST5: RUN ALL TESTS
def test5(camera):
    test2(camera)
    test3(camera)
    test4(camera)    


def parse_args():
    parser = argparse.ArgumentParser(description="Test script for OBCamera class")
    parser.add_argument("--serial_no", type=str, help="Camera serial number")
    parser.add_argument("--test", type=int, help="Test number (1-5)")
    return parser.parse_args()


if __name__ == "__main__":
    
    print("Testing OBCamera class")
    
    args = parse_args()

    if args.serial_no is None:
        print("Suggestion: Use --serial_no option to specify camera serial no")
        serial_no = input("Enter camera serial no: ")
    else:
        serial_no = args.serial_no

    camera = OBCamera(serial_no=serial_no)
    
    if args.test is not None:
        test = args.test
    else:
        print("Suggestion: Use --test option to specify test number")
        print("Available tests: ")
        print("1. Get RGB/DEPTH intrinsics")
        print("2. Get RGB image")
        print("3. Get depth image")
        print("4. Get point cloud")
        print("5. Run all tests")
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
        case _:
            print("Invalid test number")

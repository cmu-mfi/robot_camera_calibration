## Pre-requisites

* Install the Pyk4a library by following the instructions at [https://github.com/etiennedub/pyk4a](https://github.com/etiennedub/pyk4a).

## Testing the class

1. Create a venv and install the necessary requirements.
3. Run the `tests/test.py` script to test the class:
    1. **Get RGB/DEPTH intrinsics**: Retrieve the intrinsic parameters for the RGB and depth cameras using Pyk4a.
    2. **Get RGB image**: Capture and return an RGB image from the camera using Pyk4a.
    3. **Get default depth image**: Capture and return a depth image from the camera using Pyk4a.
    4. **Get default point cloud**: Generate and return a point cloud from the depth data using Pyk4a.
    5. **Visualize point cloud**: Capture and visualize a point cloud using Pyk4a.
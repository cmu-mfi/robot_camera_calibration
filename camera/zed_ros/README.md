
- `Couln't find the 'IGEV' module` 
   
   Resolved by adding the path to the `IGEV` module to the `PYTHONPATH` environment variable. 

    ```shell
    export PYTHONPATH=$PYTHONPATH:/home/mfi/repos/rtc_vision_toolbox/camera/zed_ros/IGEV/IGEV-Stereo/core
    export PYTHONPATH=$PYTHONPATH:/home/mfi/repos/rtc_vision_toolbox/camera/zed_ros/IGEV/IGEV-Stereo/
    ```

- `No such file or directory: '<path/to/IGEV/pretrained_models/middlebury/middlebury.pth>`

    - Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz?usp=share_link)
    - Unzip and copy them in the `IGEV/pretrained_models/` directory.
## Troubleshoot

* Issue: `start_streaming failed with err_code=UVC_ERROR_NO_MEM, try to increse the usbfs buffer size!` \
  Solution:   https://github.com/OpenKinect/libfreenect2/issues/807

* Issue: `Could not open device` \
  Solution: 
    ```
    sudo bash ./pyorbbecsdk/scripts/install_udev_rules.sh
    sudo udevadm control --reload-rules && sudo udevadm trigger
    ```

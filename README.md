
# go-rknnlite

go-rknnlite provides Go language bindings for the [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2/tree/master)
C API interface.  It aims to provide lite bindings in the spirit of the closed source
Python lite bindings used for running AI Inference models on the Rockchip NPU 
via the RKNN software stack.

These bindings have only been tested on the [RK3588](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html)
(specifically the Radxa Rock Pi 5B) but should work on other RK3588 based SBC's.
It should also work with other models in the RK35xx series supported by the RKNN Toolkit2.


## Usage

To use in your Go project, get the library.
```
go get github.com/swdee/go-rknnlite
```

Or to try the examples clone the git code and data repositories.  
```
git clone https://github.com/swdee/go-rknnlite.git
cd go-rknnlite/example
git clone https://github.com/swdee/go-rknnlite-data.git data
```

Then refer to the Readme files for each example to run on command line.


## Dependencies

The [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) must be installed on 
your system with C header files available in the system path, eg: `/usr/include/rknn_api.h`.

Refer to the official documentation on how to install this on your system as it
will vary based on OS and SBC vendor.

### Rock Pi 5B

My usage was on the Radxa Rock Pi 5B running the official Debian 11 OS image. 

I used the prebuilt RKNN libraries built [here](https://github.com/radxa-pkg/rknn2/releases).

```
wget https://github.com/radxa-pkg/rknn2/releases/download/1.6.0-2/rknpu2-rk3588_1.6.0-2_arm64.deb
apt install ./rknpu2-rk3588_1.6.0-2_arm64.deb 
```

### GoCV

The examples make use of [GoCV](https://gocv.io/) for image processing.  Make sure
you have a working installation of GoCV first, see the instructions in the link
for installation on your system.



## Examples

See the [example](example) directory.

* Image Classification
  * [MobileNet Demo](example/mobilenet)
  * [Pooled Runtime Usage](example/pool)
* Object Detection
  * [YOLOv5 Demo](example/yolov5)  
  * [YOLOv8 Demo](example/yolov8)
* License Plate Recognition
  * [LPRNet Demo](example/lprnet) 
  * [ALPR Demo](example/alpr) - Automatic License Plate Recognition combining Yolov8 and LPRNet Models


## Pooled Runtimes

Running multiple Runtimes in a Pool allows you to take advantage of all three
NPU cores.  For our usage of an EfficentNet-Lite0 model, a single runtime has
an inference speed of 7.9ms per image, however running a Pool of 9 runtimes brings
the average inference speed down to 1.65ms per image.

See the [Pool example](example/pool).


## Notice

This code is being used in production for Image Classification.  Over time it will be expanded
on to support more features such as Object Detection using YOLO.   The addition of
new features may cause changes or breakages in the API between commits due to the
early nature of how this library evolves.

Ensure you use Go Modules so your code is not effected, but be aware any updates may
require minor changes to your code to support the latest version.

Versioning of the library will be added at a later date once the feature set stablises.


## Post Processing

If a Model (ie: specific YOLO version) is not yet supported, a post processor 
could be written to handle the outputs from the RKNN engine in the same manner the
YOLOv5 code has been created.   


## Reference Material

* [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) - RKNN software stack
tools and C API.
* [C API Reference Documentation](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/04_Rockchip_RKNPU_API_Reference_RKNNRT_V2.0.0beta0_EN.pdf)
* [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/examples) - RKNN maintained Model Zoo with example code
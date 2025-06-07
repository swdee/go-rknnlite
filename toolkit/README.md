# RKNN Toolkit 2

## Overview

The [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) is used for compiling your 
inference models into RKNN format.   Rockchip also provides a [Model Zoo](https://github.com/airockchip/rknn_model_zoo) 
with scripts to assist this conversion.

Compiling to RKNN format is done on a x86 workstation and the Dockerfile
included in this directory has the rknn-toolkit2 and the Model Zoo files prepared with
python dependencies installed.

Along with the [compile-models.sh](compile-models.sh) script the original ONNX model files used
in the [examples](../example/) can be easily
compiled to RKNN format for your target platform: rk3562, rk3566, rk3568, rk3576, or rk3588. 

Note that rk3566 and rk3568 share the same RKNN compiled model, as does rk3582 and rk3588 are the same.


## Compile Example Models

Models compiled to RKNN format used in the [examples](../example/) are already 
available at [example/data/models/](https://github.com/swdee/go-rknnlite-data/tree/master/models), however these are generated 
as follows on a x86 workstation.

```
# run from project toolkit directory 
cd toolkit/

docker run --rm \
  -v ../example/data/models:/opt/rkmodels \
  -v "$(pwd)/compile-models.sh:/compile-models.sh" \
  swdee/rknn-toolkit:latest \
  bash -c "chmod +x /compile-models.sh && /compile-models.sh <platform>"
```

An explanation of each parameter in the docker command is as follows;

| Parameter	                                                             | Description                                                                                              |
|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| -v ../example/data/models:/opt/rkmodels                                      | Mount the directory ../example/data/models from your x86 workstation into the container as /opt/rkmodels |
| -v "$(pwd)/compile-models.sh:/compile-models.sh"                       | Copy the compile-models.sh script into the container                                                     |
| swdee/rknn-toolkit:latest                                               | Use the prebuilt rknn-toolkit docker image                                                               |
| bash -c "chmod +x /compile-models.sh && /compile-models.sh <platform>"  | Make the compile-models.sh script executable and compile models for <platform>                           |

Make sure you replace `<platform>` in the above docker command with the Rockchip platform
options of `rk3562|rk3566|rk3568|rk3576|rk3588`.  The parameter `all` can be used to
compile models for all platforms.


## Compile Custom Model


If you have trained your own model you can use the Docker image to compile to RKNN format.  

Take a custom trained YOLOv8 model for example with ONNX file located on your x86 workstation 
at `/tmp/my-model/yolov8.onnx`, we can
use the Model Zoo [convert.py](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/python/convert.py)
script to compile to RKNN format for the `rk3588` platform.

```
# run from project toolkit directory
cd toolkit/

docker run --rm \
  -v /tmp/my-model:/tmp/my-model \
  swdee/rknn-toolkit:latest \
  bash -c "cd /opt/rknn_model_zoo/examples/yolov8/python && python convert.py /tmp/my-model/yolov8.onnx rk3588 i8 /tmp/my-model/yolov8-rk3588.rknn"
```

After this command successfully runs the compiled RKNN model will be on your x86 workstation at
`/tmp/my-model/yolov8-rk3588.rknn`.  You can then copy this file to your Rockchip based SBC for running
on the NPU.




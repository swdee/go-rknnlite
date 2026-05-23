# ALPR Example

## Overview

This example demonstrates full Automatic License Plate Recognition (ALPR) by using
two models
  1. YOLOv8n - Used for license plate detection in an image
  2. LPRNet - Used for number plate recognition to read the text/characters on the number plate

The YOLOv8n model was trained on the 
[License Plate Recognition Computer Vision Project Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)
 then converted to ONNX and finally compiled to RKNN format.  You have to use
[Rockchips fork](https://github.com/airockchip/ultralytics_yolov8) of the 
Ultralytics YOLOv8 repository due to some optimizations they have made for the NPU discussed 
[here](https://github.com/rockchip-linux/rknn-toolkit2/issues/272).  

Use the `ultralytics/engine/exporter.py` script in this repository to convert the
PyTorch model to ONNX.

Then compile from ONNX to RKNN using the [conversion script](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/python/convert.py)
from the Model Zoo making sure to provide it a subset of images it was trained
on for the quantization process.

The LPRNet model used from the RKNN Model Zoo is for Chinese License Plates.
I don't think it's very good, the input size of 94x24 pixels is too small and 
the dataset it was trained on doesn't use angled number plate images.  
These factors reduce accuracy, so training your own model for real world 
usage would be better.


## Usage


Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the ALPR example on rk3588 or replace with your Platform model.
```
cd example/alpr
go run alpr.go -p rk3588
```

This will result in the output of:
```
Model first run speed: YOLO inference=22.612777ms, YOLO post processing=344.743µs, Plate recognition=7.280153ms, Plate post processing=1.060771ms, Total time=31.298736ms
Saved object detection result to ../data/car-cn-alpr-out.jpg
Benchmark count=100 warmup=5
detect: min=23.172183ms p50=28.426166ms p90=36.697384ms max=40.267315ms
total: min=23.176267ms p50=28.429957ms p90=36.705258ms max=40.274315ms
done
```

The saved JPG image with object detection markers and license plate recognised.  The 
percentage in brackets indicates the confidence score from the YOLO model for 
license plate detection.

![car-cn-out.jpg](car-cn-out.jpg)

See help for passing parameters to try your own images.
```
$ go run alpr.go -h
Usage of /tmp/go-build3203844595/b001/exe/alpr:
  -f string
        The TTF font to use (default "../data/fzhei-b01s-regular.ttf")
  -i string
        Image file to run object detection on (default "../data/car-cn.jpg")
  -l string
        RKNN compiled LPRNet model file (default "../data/models/rk3588/lprnet-rk3588.rknn")
  -m string
        RKNN compiled YOLO model file (default "../data/models/rk3588/lpd-yolov8n-rk3588.rknn")
  -o string
        The output JPG file with object detection markers (default "../data/car-cn-alpr-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
  -t string
        The text drawing mode [cn|en] (default "cn")        
```

### Docker

To run the ALPR example using the prebuilt docker image, make sure the data files have been downloaded first,
then run.
```
# from project root directory

docker run --rm \
  --device /dev/dri:/dev/dri \
  -v "$(pwd):/go/src/app" \
  -v "$(pwd)/example/data:/go/src/data" \
  -v "/usr/include/rknn_api.h:/usr/include/rknn_api.h" \
  -v "/usr/lib/librknnrt.so:/usr/lib/librknnrt.so" \
  -w /go/src/app \
  swdee/go-rknnlite:latest \
  go run ./example/alpr/alpr.go -p rk3588
```


## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Average Inference Time Per Image (p50) |
|----------|----------------------------------------|
| rk3588   | 28.4ms                                 |
| rk3576   | 35.1ms                                 |
| rk3566   | 69.8ms                                 |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available.


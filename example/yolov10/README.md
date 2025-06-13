# YOLOv10 Example


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the YOLOv10 example on rk3588 or replace with your Platform model.
```
cd example/yolov10
go run yolov10.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 6
Input tensors:
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
Output tensors:
  index=0, name=485, n_dims=4, dims=[1, 64, 80, 80], n_elems=409600, size=409600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-62, scale=0.086849
  index=1, name=499, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.002931
  index=2, name=506, n_dims=4, dims=[1, 64, 40, 40], n_elems=102400, size=102400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-55, scale=0.072764
  index=3, name=520, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003646
  index=4, name=527, n_dims=4, dims=[1, 64, 20, 20], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-68, scale=0.058066
  index=5, name=541, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003894
bus @ (92 136 555 436) 0.954108
person @ (110 234 226 536) 0.911271
person @ (212 240 285 509) 0.872328
person @ (477 233 559 521) 0.825596
person @ (80 330 123 514) 0.488516
Model first run speed: inference=47.048717ms, post processing=1.271934ms, rendering=713.403µs, total time=49.034054ms
Saved object detection result to ../data/bus-yolov10-out.jpg
Benchmark time=4.091851863s, count=100, average total time=40.918518ms
done
```

The saved JPG image with object detection markers.

![bus-out.jpg](bus-out.jpg)


To use your own RKNN compiled model and images.
```
go run yolov10.go -m <RKNN model file> -i <image file> -l <labels txt file> -o <output jpg file> -p <platform>
```

The labels file should be a text file containing the labels the Model was trained on.
It should have one label per line.



See the help for command line parameters.
```
$ go run yolov10.go --help

Usage of /tmp/go-build859033258/b001/exe/yolov10:
  -i string
        Image file to run object detection on (default "../data/bus.jpg")
  -l string
        Text file containing model labels (default "../data/coco_80_labels_list.txt")
  -m string
        RKNN compiled YOLO model file (default "../data/models/rk3588/yolov10s-rk3588.rknn")
  -o string
        The output JPG file with object detection markers (default "../data/bus-yolov10-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
```


### Docker

To run the YOLOv10 example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/yolov10/yolov10.go -p rk3588
```


## Proprietary Models

The example YOLOv10 model used has been trained on the COCO dataset so makes use
of the default Post Processor setup.  If you have trained your own Model and have
set specific Classes or want to use alternative
Box and NMS Threshold values, then initialize the `postprocess.NewYOLOv10`
with your own `YOLOv10Params`.

In the file `postprocess/yolov10.go` see function `YOLOv10COCOParams` for how to
configure your own custom parameters.



## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 4.09s          | 40.91ms                          |
| rk3576   | 3.52s          | 35.28ms                          |
| rk3566   | 11.79s         | 117.92ms                         |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available.  Secondly
the Rock 4D (rk3576) has DDR5 memory versus the Rock 5B (rk3588) with slower DDR4 memory.



## Background

This YOLOv10 example is a Go conversion of the [C API example](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov10/cpp/main.cc).



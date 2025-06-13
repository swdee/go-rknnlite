# YOLOv5 Example


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the Yolov5 example on rk3588 or replace with your Platform model.
```
cd example/yolov5
go run yolov5.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 3
Input tensors:
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
Output tensors:
  index=0, name=output0, n_dims=4, dims=[1, 255, 80, 80], n_elems=1632000, size=1632000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=1, name=343, n_dims=4, dims=[1, 255, 40, 40], n_elems=408000, size=408000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=2, name=345, n_dims=4, dims=[1, 255, 20, 20], n_elems=102000, size=102000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
person @ (210 242 282 519) 0.831880
person @ (474 229 561 522) 0.805398
person @ (114 239 207 540) 0.788235
bus @ (90 135 553 459) 0.771211
person @ (78 331 122 519) 0.413103
Model first run speed: inference=34.342902ms, post processing=181.705µs, rendering=1.119107ms, total time=35.643714ms
Saved object detection result to ../data/bus-yolov5-out.jpg
Benchmark time=2.726533944s, count=100, average total time=27.265339ms
done
```

The saved JPG image with object detection markers.

![bus-out.jpg](bus-out.jpg)



To use your own RKNN compiled model and images.
```
go run yolov5.go -m <RKNN model file> -i <image file> -l <labels txt file> -o <output jpg file> -p <platform>
```

The labels file should be a text file containing the labels the Model was trained on.
It should have one label per line.


See the help for command line parameters.
```
$ go run yolov5.go --help

Usage of /tmp/go-build3921202332/b001/exe/yolov5:
  -i string
        Image file to run object detection on (default "../data/bus.jpg")
  -l string
        Text file containing model labels (default "../data/coco_80_labels_list.txt")
  -m string
        RKNN compiled YOLO model file (default "../data/models/rk3588/yolov5s-rk3588.rknn")
  -o string
        The output JPG file with object detection markers (default "../data/bus-yolov5-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
```



### Docker

To run the YOLOv5 example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/yolov5/yolov5.go -p rk3588
```



## Proprietary Models

The example YOLOv5 model used has been trained on the COCO dataset so makes use
of the default Post Processor setup.  If you have trained your own Model and have
set specific Anchor Boxes, Classes, Strides, or want to use alternative
Box and NMS Threshold values, then initialize the `postprocess.NewYOLOv5`
with your own `YOLOv5Params`.

In the file `postprocess/yolov5.go` see function `YOLOv5COCOParams` for how to
configure your own custom parameters. 


## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 3.00s          | 30.07ms                          |
| rk3576   | 2.48s          | 24.85ms                          |
| rk3566   | 6.68s          | 66.80ms                          |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available.  Secondly
the Rock 4D (rk3576) has DDR5 memory versus the Rock 5B (rk3588) with slower DDR4 memory which
explains the faster result. 




## Background

This YOLOv5 example is a Go conversion of the [C API example](https://github.com/airockchip/rknn-toolkit2/blob/v1.6.0/rknpu2/examples/rknn_yolov5_demo/src/main.cc).


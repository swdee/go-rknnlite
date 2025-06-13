# YOLOX Example


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the YOLOX example on rk3588 or replace with your Platform model.
```
cd example/yolox
go run yolox.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 3
Input tensors:
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=1.000000
Output tensors:
  index=0, name=output, n_dims=4, dims=[1, 85, 80, 80], n_elems=544000, size=544000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-28, scale=0.022949
  index=1, name=788, n_dims=4, dims=[1, 85, 40, 40], n_elems=136000, size=136000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-26, scale=0.024599
  index=2, name=output.1, n_dims=4, dims=[1, 85, 20, 20], n_elems=34000, size=34000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-19, scale=0.021201
bus @ (87 137 550 428) 0.929565
person @ (103 237 223 535) 0.895541
person @ (210 235 286 513) 0.871337
person @ (474 235 559 519) 0.830675
person @ (80 328 118 516) 0.499204
Model first run speed: inference=39.329023ms, post processing=88.081µs, rendering=689.195µs, total time=40.106299ms
Saved object detection result to ../data/bus-yolox-out.jpg
Benchmark time=3.370110888s, count=100, average total time=33.701108ms
done
```

The saved JPG image with object detection markers.

![bus-out.jpg](bus-out.jpg)


To use your own RKNN compiled model and images.
```
go run yolox.go -m <RKNN model file> -i <image file> -l <labels txt file> -o <output jpg file> -p <platform>
```

The labels file should be a text file containing the labels the Model was trained on.
It should have one label per line.


See the help for command line parameters.
```
$ go run yolox.go --help

Usage of /tmp/go-build2416613122/b001/exe/yolox:
  -i string
        Image file to run object detection on (default "../data/bus.jpg")
  -l string
        Text file containing model labels (default "../data/coco_80_labels_list.txt")
  -m string
        RKNN compiled YOLO model file (default "../data/models/rk3588/yoloxs-rk3588.rknn")
  -o string
        The output JPG file with object detection markers (default "../data/bus-yolox-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
```

### Docker

To run the YOLOX example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/yolox/yolox.go -p rk3588
```



## Proprietary Models

The example YOLOX model used has been trained on the COCO dataset so makes use
of the default Post Processor setup.  If you have trained your own Model and have
set specific Classes, Strides, or want to use alternative
Box and NMS Threshold values, then initialize the `postprocess.NewYOLOX`
with your own `YOLOXParams`.

In the file `postprocess/yolox.go` see function `YOLOXCOCOParams` for how to
configure your own custom parameters.


## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 3.37s          | 33.70ms                          |
| rk3576   | 3.29s          | 32.98ms                          |
| rk3566   | 7.61s          | 76.11ms                          |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available.  Secondly
the Rock 4D (rk3576) has DDR5 memory versus the Rock 5B (rk3588) with slower DDR4 memory.




## Background

This YOLOX example is a Go conversion of the [C API example](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolox/cpp/main.cc).


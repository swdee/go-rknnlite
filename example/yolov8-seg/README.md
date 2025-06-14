# YOLOv8-seg Example

This demo uses a YOLOv8-seg model to detect objects and provides
instance segmentation.


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the YOLOv8-seg example on rk3588 or replace with your Platform model.
```
cd example/yolov8-seg
go run yolov8-seg.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 13
Input tensors:
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
Output tensors:
  index=0, name=375, n_dims=4, dims=[1, 64, 80, 80], n_elems=409600, size=409600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-61, scale=0.115401
  index=1, name=onnx::ReduceSum_383, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003514
  index=2, name=388, n_dims=4, dims=[1, 1, 80, 80], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003540
  index=3, name=354, n_dims=4, dims=[1, 32, 80, 80], n_elems=204800, size=204800, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=27, scale=0.019863
  index=4, name=395, n_dims=4, dims=[1, 64, 40, 40], n_elems=102400, size=102400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-15, scale=0.099555
  index=5, name=onnx::ReduceSum_403, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003555
  index=6, name=407, n_dims=4, dims=[1, 1, 40, 40], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003680
  index=7, name=361, n_dims=4, dims=[1, 32, 40, 40], n_elems=51200, size=51200, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=30, scale=0.022367
  index=8, name=414, n_dims=4, dims=[1, 64, 20, 20], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-55, scale=0.074253
  index=9, name=onnx::ReduceSum_422, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003813
  index=10, name=426, n_dims=4, dims=[1, 1, 20, 20], n_elems=400, size=400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=11, name=368, n_dims=4, dims=[1, 32, 20, 20], n_elems=12800, size=12800, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=43, scale=0.019919
  index=12, name=347, n_dims=4, dims=[1, 32, 160, 160], n_elems=819200, size=819200, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-119, scale=0.032336
Model first run speed: inference=53.334871ms, post processing=47.551235ms, rendering=1.356806ms, total time=102.242912ms
cat @ (712 98 903 332) 0.918907
cat @ (24 117 200 291) 0.860259
dog @ (173 89 359 296) 0.857901
dog @ (311 98 528 312) 0.831211
cat @ (523 142 719 299) 0.789163
Saved object detection result to ../data/catdog-yolov8-seg-out.jpg
Benchmark time=8.332262773s, count=100, average total time=83.322627ms
done
```

The saved JPG image with instance segmentation outlines.

![catdog-outline.jpg](catdog-outline.jpg)


See the help for command line parameters.
```
$ go run yolov8-seg.go --help

Usage of /tmp/go-build652765382/b001/exe/yolov8-seg:
  -i string
        Image file to run object detection on (default "../data/catdog.jpg")
  -l string
        Text file containing model labels (default "../data/coco_80_labels_list.txt")
  -m string
        RKNN compiled YOLO model file (default "../data/models/rk3588/yolov8s-seg-rk3588.rknn")
  -o string
        The output JPG file with object detection markers (default "../data/catdog-yolov8-seg-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
  -r string
        The rendering format used for instance segmentation [outline|mask|dump] (default "outline")
```



### Docker

To run the YOLOv8-seg example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/yolov8-seg/yolov8-seg.go -p rk3588
```



## Rendering Methods

The default rendering method is to draw an outline around the edge of the detected
object as depicted in the image above.   This method however takes the most
resources to calculate and is more noticable if the scene has more objects in it.

A faster method of rendering is also provided which draws the bounding boxes around
the object and provides a single transparent overlay to indicate the segment mask.

This can be output with the following flag.
```
go run yolov8-seg.go -p rk3588 -r mask
```

![catdog-mask.jpg](catdog-mask.jpg)

For visualisation and debugging purposes the segmentation mask can also be dumped
to an image.
```
go run yolov8-seg.go -p rk3588 -r dump
```

![catdog-dump.jpg](catdog-dump.jpg)


## Model Segment Mask Size

The Rockchip examples are based on Models with an input tensor size of 640x640 with
3 channels (RGB).  The output tensor for the segment mask size is 160x160.

If your Model has been trained with a different output segment mask size such as 320x320
you will need to pass those sizes to the `YOLOv8SegParams.Prototype*` variables, eg:

```
// start with default COCO Parameters
yParams := postprocess.YOLOv8SegCOCOParams()

// Set your Models output mask size
yParams.PrototypeHeight = 320
yParams.PrototypeWeight = 320

// create YOLO Processor instance	
yoloProcesser := postprocess.NewYOLOv8Seg(yParams)
```


## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 8.33s          | 83.32ms                          |
| rk3576   | 10.23s         | 102.33ms                         |
| rk3566   | 36.02s         | 360.23ms                         |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available.  Secondly
the Rock 4D (rk3576) has DDR5 memory versus the Rock 5B (rk3588) with slower DDR4 memory.




## Background

This YOLOv8-seg example is a Go conversion of the [C API example](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8_seg/cpp/main.cc)
with improvements made to it inspired by [Ultralytics Instance Segmentation](https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/#what-is-instance-segmentation).


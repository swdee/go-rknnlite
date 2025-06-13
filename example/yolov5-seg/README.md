# YOLOv5-seg Example

This demo uses a YOLOv5-seg model to detect objects and provides 
instance segmentation. 


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the YOLOv5-seg example on rk3588 or replace with your Platform model.
```
cd example/yolov5-seg
go run yolov5-seg.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 7
Input tensors:
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
Output tensors:
  index=0, name=output0, n_dims=4, dims=[1, 255, 80, 80], n_elems=1632000, size=1632000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=1, name=output1, n_dims=4, dims=[1, 96, 80, 80], n_elems=614400, size=614400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=20, scale=0.022222
  index=2, name=376, n_dims=4, dims=[1, 255, 40, 40], n_elems=408000, size=408000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=3, name=377, n_dims=4, dims=[1, 96, 40, 40], n_elems=153600, size=153600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=29, scale=0.023239
  index=4, name=379, n_dims=4, dims=[1, 255, 20, 20], n_elems=102000, size=102000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003918
  index=5, name=380, n_dims=4, dims=[1, 96, 20, 20], n_elems=38400, size=38400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=32, scale=0.024074
  index=6, name=371, n_dims=4, dims=[1, 32, 160, 160], n_elems=819200, size=819200, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-116, scale=0.022475
dog @ (197 83 357 299) 0.786010
cat @ (714 101 900 336) 0.706588
dog @ (312 93 526 304) 0.693387
cat @ (28 113 171 292) 0.641764
cat @ (530 141 712 299) 0.616804
Model first run speed: inference=45.977719ms, post processing=46.101967ms, rendering=1.395305ms, total time=93.474991ms
Saved object detection result to ../data/catdog-yolov5-seg-out.jpg
Benchmark time=7.346591785s, count=100, average total time=73.465917ms
done
```

The saved JPG image with instance segmentation outlines.

![catdog-outline.jpg](catdog-outline.jpg)


See the help for command line parameters.
```
$ go run yolov5-seg.go --help

Usage of /tmp/go-build2169893350/b001/exe/yolov5-seg:
  -i string
        Image file to run object detection on (default "../data/catdog.jpg")
  -l string
        Text file containing model labels (default "../data/coco_80_labels_list.txt")
  -m string
        RKNN compiled YOLO model file (default "../data/yolov5s-seg-rk3588.rknn")
  -o string
        The output JPG file with object detection markers (default "../data/catdog-yolov5-seg-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
  -r string
        The rendering format used for instance segmentation [outline|mask|dump] (default "outline")
```

### Docker

To run the YOLOv5-seg example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/yolov5-seg/yolov5-seg.go -p rk3588
```




## Rendering Methods

The default rendering method is to draw an outline around the edge of the detected
object as depicted in the image above.   This method however takes the most
resources to calculate and is more noticable if the scene has more objects in it.

A faster method of rendering is also provided which draws the bounding boxes around
the object and provides a single transparent overlay to indicate the segment mask.

This can be output with the following flag.
```
go run yolov5-seg.go -p rk3588 -r mask
```

![catdog-mask.jpg](catdog-mask.jpg)

For visualisation and debugging purposes the segmentation mask can also be dumped
to an image.
```
go run yolov5-seg.go -p rk3588 -r dump
```

![catdog-dump.jpg](catdog-dump.jpg)


## Model Segment Mask Size

The Rockchip examples are based on Models with an input tensor size of 640x640 with
3 channels (RGB).  The output tensor for the segment mask size is 160x160.

If your Model has been trained with a different output segment mask size such as 320x320
you will need to pass those sizes to the `YOLOv8SegParams.Prototype*` variables, eg:

```
// start with default COCO Parameters
yParams := postprocess.YOLOv5SegCOCOParams()

// Set your Models output mask size
yParams.PrototypeHeight = 320
yParams.PrototypeWeight = 320

// create YOLO Processor instance	
yoloProcesser := postprocess.NewYOLOv5Seg(yParams)
```



## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 7.34s          | 73.46ms                          |
| rk3576   | 8.91s          | 89.14ms                          |
| rk3566   | 32.64s         | 326.44ms                         |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available.  Secondly
the Rock 4D (rk3576) has DDR5 memory versus the Rock 5B (rk3588) with slower DDR4 memory.



## Background

This YOLOv5-seg example is a Go conversion of the [C API example](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov5_seg/cpp/main.cc)
with improvements made to it inspired by [Ultralytics Instance Segmentation](https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/#what-is-instance-segmentation).


# YOLOv11 Example


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone https://github.com/swdee/go-rknnlite-data.git data
```

Run the YOLOv11 example.
```
cd example/yolov11
go run yolov11.go
```

This will result in the output of:
```
Driver Version: 0.8.2, API Version: 1.6.0 (9a7b5d24c@2023-12-13T17:31:11)
Model Input Number: 1, Ouput Number: 9
Input tensors:
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
Output tensors:
  index=0, name=462, n_dims=4, dims=[1, 64, 80, 80], n_elems=409600, size=409600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-60, scale=0.117048
  index=1, name=onnx::ReduceSum_476, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003011
  index=2, name=480, n_dims=4, dims=[1, 1, 80, 80], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003041
  index=3, name=487, n_dims=4, dims=[1, 64, 40, 40], n_elems=102400, size=102400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-37, scale=0.084344
  index=4, name=onnx::ReduceSum_501, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003592
  index=5, name=505, n_dims=4, dims=[1, 1, 40, 40], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=6, name=512, n_dims=4, dims=[1, 64, 20, 20], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-44, scale=0.078218
  index=7, name=onnx::ReduceSum_526, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003861
  index=8, name=530, n_dims=4, dims=[1, 1, 20, 20], n_elems=400, size=400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003878
bus @ (94 134 551 437) 0.945933
person @ (475 231 559 522) 0.895741
person @ (108 233 227 535) 0.891880
person @ (211 240 284 508) 0.888019
person @ (79 325 126 518) 0.606169
Model first run speed: inference=62.703784ms, post processing=1.303424ms, rendering=1.429421ms, total time=65.436629ms
Saved object detection result to ../data/bus-yolov11-out.jpg
Benchmark time=4.206316731s, count=100, average total time=42.063167ms
done
```

The saved JPG image with object detection markers.

![bus-out.jpg](bus-out.jpg)


To use your own RKNN compiled model and images.
```
go run yolov11.go -m <RKNN model file> -i <image file> -l <labels txt file> -o <output jpg file>
```

The labels file should be a text file containing the labels the Model was trained on.
It should have one label per line.




### Docker

To run the YOLOv11 example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/yolov11/yolov11.go
```



## Proprietary Models

The example YOLOv11 model used has been trained on the COCO dataset so makes use
of the default Post Processor setup.  If you have trained your own Model and have
set specific Classes or want to use alternative
Box and NMS Threshold values, then initialize the `postprocess.NewYOLOv11`
with your own `YOLOv11Params`.

In the file `postprocess/yolov11.go` see function `YOLOv11COCOParams` for how to
configure your own custom parameters.


## Background

This YOLOv11 example is a Go conversion of the [C API example](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolo11/cpp/main.cc).



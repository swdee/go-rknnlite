# RetinaFace Example

## Overview


Detect human faces in images and identify face landmark features (eyes, nose, and mouth).


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the RetinaFace example on rk3588 or replace with your Platform model.
```
cd example/retinaface
go run retinaface.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 3
Input tensors:
  index=0, name=input0, n_dims=4, dims=[1, 320, 320, 3], n_elems=307200, size=307200, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-14, scale=1.074510
Output tensors:
  index=0, name=output0, n_dims=3, dims=[1, 4200, 4, 0], n_elems=16800, size=16800, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=0, scale=0.044699
  index=1, name=572, n_dims=3, dims=[1, 4200, 2, 0], n_elems=8400, size=16800, fmt=UNDEFINED, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000
  index=2, name=571, n_dims=3, dims=[1, 4200, 10, 0], n_elems=42000, size=42000, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=-22, scale=0.086195
Model first run speed: inference=9.732158ms, post processing=163.914µs, rendering=1.654012ms, total time=11.550084ms
face @ (312 531 453 714) 0.998047
face @ (306 289 436 454) 0.997559
face @ (53 286 184 460) 0.996582
face @ (543 533 680 716) 0.996094
face @ (56 537 181 703) 0.995605
face @ (61 34 192 209) 0.994629
face @ (523 274 674 474) 0.994141
face @ (553 28 695 224) 0.991211
face @ (292 36 421 217) 0.991211
Saved object detection result to ../data/face-out.jpg
Benchmark time=573.784348ms, count=100, average total time=5.737843ms
done
```

The saved JPG image with face landmarks indicated.

![face-out.jpg](face-out.jpg)



See the help for command line parameters.
```
$ go run retinaface.go --help

Usage of /tmp/go-build3703609935/b001/exe/retinaface:
  -i string
        Image file to run inference on (default "../data/face.jpg")
  -m string
        RKNN compiled Retina Face model file (default "../data/models/rk3588/retinaface-320-rk3588.rknn")
  -o string
        The output JPG file with face detection markers (default "../data/face-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
```


### Docker

To run the RetinaFace example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/retinaface/retinaface.go -p rk3588
```


## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 0.57s          | 5.73ms                           |
| rk3576   | 0.52s          | 5.27ms                           |
| rk3566   | 1.12s          | 11.21ms                          |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available. Secondly
the Rock 4D (rk3576) has DDR5 memory versus the Rock 5B (rk3588) with slower DDR4 memory.



## Background

This RetinaFace example is a Go conversion of the [C API example](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/RetinaFace/cpp/main.cc).




# YOLOv8-pose Example


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the YOLOv8-pose example on rk3588 or replace with your Platform model.
```
cd example/yolov8-pose
go run yolov8-pose.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 4
Input tensors:
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
Output tensors:
  index=0, name=/model.22/Concat_1_output_0, n_dims=4, dims=[1, 65, 80, 80], n_elems=416000, size=416000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=31, scale=0.141366
  index=1, name=/model.22/Concat_2_output_0, n_dims=4, dims=[1, 65, 40, 40], n_elems=104000, size=104000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=67, scale=0.235531
  index=2, name=/model.22/Concat_3_output_0, n_dims=4, dims=[1, 65, 20, 20], n_elems=26000, size=26000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=50, scale=0.159995
  index=3, name=/model.22/Concat_6_output_0, n_dims=4, dims=[1, 17, 3, 8400], n_elems=428400, size=856800, fmt=NCHW, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000
person @ (49 104 168 526) 0.903778
person @ (348 94 486 516) 0.903778
person @ (472 116 579 525) 0.903778
person @ (251 143 363 520) 0.872132
person @ (678 112 779 526) 0.872132
person @ (161 109 266 521) 0.872132
person @ (572 103 686 528) 0.853203
Model first run speed: inference=37.624844ms, post processing=465.199µs, rendering=1.815006ms, total time=39.905049ms
Saved object detection result to ../data/people-yolov8-pose-out.jpg
Benchmark time=2.860705252s, count=100, average total time=28.607052ms
done
```

The saved JPG image with pose estimation markers.

![people-out.jpg](people-out.jpg)



See the help for command line parameters.
```
$ go run yolov8-pose.go --help

Usage of /tmp/go-build2381700544/b001/exe/yolov8-pose:
  -i string
        Image file to run object detection on (default "../data/people-poses.jpg")
  -l string
        Text file containing model labels (default "../data/yolov8_pose_labels_list.txt")
  -m string
        RKNN compiled YOLO model file (default "../data/models/rk3588/yolov8n-pose-rk3588.rknn")
  -o string
        The output JPG file with pose detection markers (default "../data/people-yolov8-pose-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
```



### Docker

To run the YOLOv8-pose example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/yolov8-pose/yolov8-pose.go -p rk3588
```

## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 2.86s          | 28.60ms                          |
| rk3576   | 3.64s          | 36.49ms                          |
| rk3566   | 5.48s          | 54.84ms                          |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available.  Secondly
the Rock 4D (rk3576) has DDR5 memory versus the Rock 5B (rk3588) with slower DDR4 memory.




## Background

This YOLOv8-pose example is a Go conversion of the [C API example](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8_pose/cpp/main.cc).


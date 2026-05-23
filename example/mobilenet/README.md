# MobileNet Example

## Usage

To run the MobileNet example make sure you have downloaded the data files first.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the MobileNet example on rk3588 or replace with your Platform model.
```
cd example/mobilenet/
go run mobilenet.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.8.2, API Version: 1.6.0 (9a7b5d24c@2023-12-13T17:31:11)
Model Input Number: 1, Ouput Number: 1
Input tensors:
  index=0, name=input, n_dims=4, dims=[1, 224, 224, 3], n_elems=150528, size=150528, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=0, scale=0.007812
Output tensors:
  index=0, name=MobilenetV1/Predictions/Reshape_1, n_dims=2, dims=[1, 1001, 0, 0], n_elems=1001, size=1001, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003906
 --- Top5 ---
283: 0.468750
282: 0.242188
286: 0.105469
464: 0.089844
264: 0.019531
Benchmark count=100 warmup=5
inference: min=1.941296ms p50=1.944504ms p90=1.962295ms max=2.820945ms
postprocess: min=44.332µs p50=44.624µs p90=44.625µs max=70.291µs
total: min=1.986795ms p50=1.990295ms p90=2.011003ms max=2.88657ms
done
```

To use your own RKNN compiled model and images.
```
go run mobilenet.go -m <RKNN model file> -i <image file> -p <platform>
```

See the help for command line parameters.
```
$ go run mobilenet.go --help

Usage of /tmp/go-build2453632432/b001/exe/mobilenet:
  -i string
        Image file to run inference on (default "../data/cat_224x224.jpg")
  -m string
        RKNN compiled model file (default "../data/models/rk3588/mobilenet_v1-rk3588.rknn")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
```

### Docker

To run the MobileNet example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/mobilenet/mobilenet.go -p rk3588
```


## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Average Inference Time Per Image (p50) |
|----------|----------------------------------------|
| rk3588   | 1.99ms                                 |
| rk3576   | 2.52ms                                 |
| rk3566   | 4.88ms                                 |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available.  Secondly
the Rock 4D (rk3576) has DDR5 memory versus the Rock 5B (rk3588) with slower DDR4 memory which
explains the faster result.


## Background

This MobileNet example is a Go conversion of the [C API example](https://github.com/airockchip/rknn-toolkit2/blob/v1.6.0/rknpu2/examples/rknn_mobilenet_demo/src/main.cc).


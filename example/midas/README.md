# MiDaS Depth Estimation Example

## Overview

This example uses the [MiDaS v3.1 depth estimation](https://github.com/isl-org/MiDaS/)
for computing depth in a single image.


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

![bedroom.jpg](https://github.com/swdee/go-rknnlite-data/raw/master/bedroom.jpg)

Run the MiDaS example on the above living room scene on rk3588 or replace with your Platform model.
```
cd example/midas
go run midas.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Output Number: 1
Input tensors:
  index=0, name=input, n_dims=4, dims=[1, 256, 256, 3], n_elems=196608, size=196608, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=0, scale=0.007843
Output tensors:
  index=0, name=depth, n_dims=4, dims=[1, 1, 256, 256], n_elems=65536, size=65536, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=19.864582
Model first run speed: inference=577.442314ms, post processing=1.180646ms, rendering=3.137693ms, total time=581.760653ms
Saved depth map result to ../data/bedroom-out.jpg
Benchmark time=12.167970519s, count=20, average total time=608.398525ms
done
```

The saved JPG image with depth estimation map.

![midas-bedroom-out.jpg](https://github.com/swdee/go-rknnlite-data/raw/master/docimg/midas-bedroom-out.jpg)


See the help for command line parameters.
```
$ go run midas.go -h

Usage of /tmp/go-build2937772053/b001/exe/midas:
  -i string
        Image file to run depth estimation on (default "../data/bedroom.jpg")
  -m string
        RKNN compiled depth model file (default "../data/models/rk3588/dpt_swin2_tiny_256-rk3588.rknn")
  -o string
        Output JPG file (depth visualization) (default "../data/bedroom-out.jpg")
  -p string
        Rockchip platform [rk3562|rk3566|rk3568|rk3576|rk3582|rk3588] (default "rk3588")
```




### Docker

To run the MiDaS example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/midas/midas.go -p rk3588
```


## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.

| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 12.16s         | 608.39ms                         |
| rk3576   | 16.85s         | 842.97ms                         |
| rk3566   | 37.49s         | 1.87s                            


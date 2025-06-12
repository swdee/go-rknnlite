# LPRNet Example

## Usage


Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the LPRNet example on rk3588 or replace with your Platform model.
```
cd example/lprnet
go run lprnet.go -p rk3588
```


This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 1
Input tensors:
  index=0, name=input, n_dims=4, dims=[1, 24, 94, 3], n_elems=6768, size=6768, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=0, scale=0.007843
Output tensors:
  index=0, name=output, n_dims=3, dims=[1, 68, 18, 0], n_elems=1224, size=1224, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=50, scale=0.643529
Model first run speed: inference=4.203128ms, post processing=30.916µs, total time=4.234044ms
License plate recognition result: 湘F6CL03
Benchmark time=350.625899ms, count=100, average total time=3.506258ms
done
```

To use your own RKNN compiled model and images.
```
go run lprnet.go -m <RKNN model file> -i <image file> -p <platform>
```


See the help for command line parameters.
```
$ go run lprnet.go --help

Usage of /tmp/go-build233788912/b001/exe/lprnet:
  -i string
        Image file to run inference on (default "../data/lplate.jpg")
  -m string
        RKNN compiled model file (default "../data/models/rk3588/lprnet-rk3588.rknn")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
```


### Docker

To run the ALPR example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/lprnet/lprnet.go -p rk3588
```


## Proprietary Models

This example makes use of the [Chinese License Plate Recognition LPRNet](https://github.com/sirius-ai/LPRNet_Pytorch). 
You can train your own LPRNet's for other countries but need to initialize
the `postprocess.NewLPRNet` with your specific `LPRNetParams` containing the
maximum length of your countries number plates and character set used.



## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 0.35s          | 3.50ms                           |
| rk3576   | 0.49s          | 4.96ms                           |
| rk3566   | 1.63s          | 16.32ms                          |

Note that these examples are only using a single NPU core to run inference on.  The results
would be different when running a Pool of models using all NPU cores available.



## Background

This LPRNet example is a Go conversion of the [C API Example](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/LPRNet/cpp/main.cc)


## References

* [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/pdf/1806.10447v1) - Original
paper proposing LPRNet.
* [An End to End Recognition for License Plates Using Convolutional Neural Networks](https://www.researchgate.net/publication/332650352_An_End_to_End_Recognition_for_License_Plates_Using_Convolutional_Neural_Networks) - A paper
that looks at LPRNet usage specific to number plates used in China.
* [Automatic License Plate Recognition](https://hailo.ai/blog/automatic-license-plate-recognition-with-hailo-8/) - An overview
of creating a full ALPR architecture that uses; Vehicle detection (YOLO),  License Plate Detection (LPDNet), 
and License Plate Recognition (LPRNet). 

# LPRNet Example

## Usage


Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone https://github.com/swdee/go-rknnlite-data.git data
```

Run the LPRNet example.
```
cd example/lprnet
go run lprnet.go
```


This will result in the output of:
```
Driver Version: 0.8.2, API Version: 1.6.0 (9a7b5d24c@2023-12-13T17:31:11)
Model Input Number: 1, Ouput Number: 1
Input tensors:
  index=0, name=input, n_dims=4, dims=[1, 24, 94, 3], n_elems=6768, size=6768, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=0, scale=0.007843
Output tensors:
  index=0, name=output, n_dims=3, dims=[1, 68, 18, 0], n_elems=1224, size=1224, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=47, scale=0.911201
Model first run speed: inference=7.787585ms, post processing=25.374µs, total time=7.812959ms
License plate recognition result: 湘F6CL03
Benchmark time=61.070751ms, count=10, average total time=6.107075ms
done
```

To use your own RKNN compiled model and images.
```
go run lprnet.go -m <RKNN model file> -i <image file>
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
  go run ./example/lprnet/lprnet.go
```


## Proprietary Models

This example makes use of the [Chinese License Plate Recognition LPRNet](https://github.com/sirius-ai/LPRNet_Pytorch). 
You can train your own LPRNet's for other countries but need to initialize
the `postprocess.NewLPRNet` with your specific `LPRNetParams` containing the
maximum length of your countries number plates and character set used.


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

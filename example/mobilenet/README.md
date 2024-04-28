# MobileNet Example

## Usage

To run the MobileNet example make sure you have downloaded the data files first.
You only need to do this once for all examples.

```
cd example/
git clone https://github.com/swdee/go-rknnlite-data.git data
```

Run the MobileNet example.
```
cd example/mobilenet/
go run mobilenet.go 
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
done
```

To use your own RKNN compiled model and images.
```
go run mobilenet.go -m <RKNN model file> -i <image file>
```

## Background

This MobileNet example is a Go conversion of the [C API example](https://github.com/airockchip/rknn-toolkit2/blob/v1.6.0/rknpu2/examples/rknn_mobilenet_demo/src/main.cc).

